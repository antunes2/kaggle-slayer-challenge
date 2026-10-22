"""External data (D0009): public Kaggle datasets referenced by top notebooks.

Flow (all code, fail-soft): the notebook digest's `[EXTERNAL]` lines → dataset refs →
size-capped CSV download into `workspace/external/` → `external/manifest.json` (source,
files, rows, columns) → a compact summary for the planner and researcher. The harness
hands the frames to the pipeline's optional hooks; external rows only ever join
training folds, so CV stays honest.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from slayer.loop.engine import RunContext
from slayer.workspace import EventType

MAX_DATASETS = 2
MAX_TOTAL_MB = 500
MAX_SUMMARY_COLUMNS = 40

_OWNER_PATH = re.compile(r"/kaggle/input/datasets/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")
_BARE_PATH = re.compile(r"/kaggle/input/([A-Za-z0-9_.-]+)/")


def external_enabled(policy: str, category: str | None) -> bool:
    """Resolve the run's external-data policy: on / off / auto (= Playground only)."""
    if policy == "on":
        return True
    if policy == "off":
        return False
    return (category or "").strip().lower() == "playground"


def dataset_refs_from_digest(digest_md: str, competition: str) -> list[str]:
    """Dataset refs seen in `[EXTERNAL]` digest lines, most-mentioned first.

    `owner/slug` refs come from `/kaggle/input/datasets/<owner>/<slug>/...`; bare
    `/kaggle/input/<slug>/...` paths yield `?/<slug>` (to be resolved by search).
    """
    counts: Counter[str] = Counter()
    for line in digest_md.splitlines():
        if not line.startswith("[EXTERNAL]"):
            continue
        for owner, slug in _OWNER_PATH.findall(line):
            counts[f"{owner}/{slug}"] += 1
        for slug in _BARE_PATH.findall(line):
            if slug in ("datasets", "competitions") or slug == competition:
                continue
            counts[f"?/{slug}"] += 1
    resolved_slugs = {ref.split("/", 1)[1] for ref in counts if not ref.startswith("?/")}
    return [
        ref for ref, _ in counts.most_common()
        if not (ref.startswith("?/") and ref[2:] in resolved_slugs)
    ]


def _manifest_path(ctx: RunContext) -> Path:
    """Location of the external-data manifest."""
    return ctx.run.workspace("external") / "manifest.json"


def load_manifest(ctx: RunContext) -> dict | None:
    """The external-data manifest, if this run prepared any external data."""
    path = _manifest_path(ctx)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _file_key(slug: str, name: str, taken: set[str]) -> str:
    """Short dict key for a file (its stem), disambiguated by dataset slug on clashes."""
    key = Path(name).stem
    if key in taken:
        key = f"{slug}__{key}"
    taken.add(key)
    return key


def prepare_external_data(ctx: RunContext, digest_md: str) -> dict | None:
    """Download up to MAX_DATASETS referenced public datasets (CSV, size-capped).

    Idempotent (an existing manifest is reused) and fail-soft (errors are journaled;
    the run continues without external data).
    """
    reports = ctx.run.workspace("reports")
    competition = json.loads((reports / "competition.json").read_text(encoding="utf-8"))
    if not external_enabled(ctx.run.external_data, competition.get("category")):
        return None
    existing = load_manifest(ctx)
    if existing is not None:
        return existing
    kaggle = ctx.kaggle
    if kaggle is None or not hasattr(kaggle, "download_dataset"):
        return None

    datasets, errors, taken = [], [], set()
    budget = MAX_TOTAL_MB * 1024 * 1024
    for ref in dataset_refs_from_digest(digest_md, ctx.run.competition):
        if len(datasets) >= MAX_DATASETS:
            break
        try:
            if ref.startswith("?/"):
                found = kaggle.find_dataset(ref[2:])
                if not found or any(d["ref"] == found for d in datasets):
                    continue
                ref = found
            files = [f for f in kaggle.dataset_files(ref) if f.name.lower().endswith(".csv")]
            size = sum(f.total_bytes or 0 for f in files)
            if not files or size > budget:
                errors.append(f"{ref}: skipped ({len(files)} csv files, {size // 2**20} MB)")
                continue
            owner, slug = ref.split("/", 1)
            target = ctx.run.workspace("external") / f"{owner}__{slug}"
            kaggle.download_dataset(ref, target)
            entries = []
            for path in sorted(target.rglob("*.csv")):
                frame = pd.read_csv(path, nrows=5)
                rows = sum(1 for _ in path.open(encoding="utf-8", errors="replace")) - 1
                entries.append({
                    "key": _file_key(slug, path.name, taken),
                    "path": str(path.resolve()),
                    "rows": rows,
                    "columns": [str(c) for c in frame.columns],
                })
            for other in target.rglob("*"):
                if other.is_file() and other.suffix.lower() != ".csv":
                    other.unlink()
            if entries:
                datasets.append({"ref": ref, "files": entries})
                budget -= size
            else:
                shutil.rmtree(target, ignore_errors=True)
        except Exception as exc:
            errors.append(f"{ref}: {str(exc)[:160]}")

    manifest = {"datasets": datasets, "errors": errors}
    if datasets:
        _manifest_path(ctx).parent.mkdir(parents=True, exist_ok=True)
        _manifest_path(ctx).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ctx.journal.append(
        agent="scout",
        type=EventType.EXTERNAL_DATA,
        data={"datasets": [d["ref"] for d in datasets], "errors": errors},
        run=ctx.run.run_id,
    )
    ctx.say(
        "  external data: " + (", ".join(d["ref"] for d in datasets) if datasets
                               else "none usable")
        + (f" ({len(errors)} skipped)" if errors else "")
    )
    return manifest if datasets else None


def external_summary_md(manifest: dict | None, target_column: str | None = None) -> str:
    """Compact description of the available external frames for agent prompts."""
    if not manifest or not manifest.get("datasets"):
        return ""
    lines = [
        "External data available (public Kaggle datasets; credited in the run manifest).",
        "The harness passes them as `external: dict[str, pandas.DataFrame]` keyed as below.",
    ]
    for dataset in manifest["datasets"]:
        for entry in dataset["files"]:
            columns = entry["columns"][:MAX_SUMMARY_COLUMNS]
            more = len(entry["columns"]) - len(columns)
            has_target = target_column is not None and target_column in entry["columns"]
            lines.append(
                f"- `external[\"{entry['key']}\"]` ← {dataset['ref']} "
                f"({entry['rows']:,} rows; target column `{target_column}` "
                f"{'present' if has_target else 'absent — check for a renamed label'}): "
                + ", ".join(columns) + (f", … (+{more})" if more > 0 else "")
            )
    return "\n".join(lines)
