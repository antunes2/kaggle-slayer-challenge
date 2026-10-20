from slayer.analysis.models import DatasetProfile
from slayer.schemas import DataAnalystOutput


def render_profile_md(profile: DatasetProfile) -> str:
    lines = [f"# Dataset profile — {profile.competition}", ""]
    lines.append(f"- train: {profile.train_rows:,} rows x {profile.train_cols} columns")
    lines.append(f"- test: {profile.test_rows:,} rows")
    lines.append(f"- target: `{profile.target_column}`")
    lines.append(f"- problem type guess: **{profile.problem_type_guess}**")
    lines.append("")
    lines.append("## Target")
    for key, value in profile.target_summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Columns")
    lines.append("| column | dtype | missing % | unique | facts |")
    lines.append("|---|---|---|---|---|")
    for col in profile.columns[:40]:
        facts = "; ".join(f"{k}={v}" for k, v in col.facts.items())
        facts = facts.replace("|", "\\|")
        lines.append(f"| {col.name} | {col.dtype} | {col.missing_pct} | {col.nunique} | {facts} |")
    if len(profile.columns) > 40:
        lines.append(f"| ... | {len(profile.columns) - 40} more columns omitted | | | |")
    lines.append("")
    if profile.top_correlations:
        lines.append("## Correlations with target (top, by |corr|)")
        for name, corr in profile.top_correlations:
            lines.append(f"- {name}: {corr:+.4f}")
        lines.append("")
    if profile.notes:
        lines.append("## Notes")
        for note in profile.notes:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def render_data_report_md(output: DataAnalystOutput) -> str:
    report = output.report
    lines = [f"# Data report — {report.problem_type} on `{report.target_column}`", ""]
    lines.append(report.summary)
    lines.append("")
    if report.key_findings:
        lines.append("## Key findings")
        lines.extend(f"- {f}" for f in report.key_findings)
        lines.append("")
    if report.data_quality_issues:
        lines.append("## Data quality issues")
        lines.extend(f"- {f}" for f in report.data_quality_issues)
        lines.append("")
    if report.risks:
        lines.append("## Risks")
        lines.extend(f"- {f}" for f in report.risks)
        lines.append("")
    if output.feature_plan.ideas:
        lines.append("## Feature plan")
        lines.append("| idea | expected | columns | rationale |")
        lines.append("|---|---|---|---|")
        for idea in output.feature_plan.ideas:
            columns = ", ".join(idea.based_on_columns)
            rationale = idea.rationale.replace("|", "\\|")
            lines.append(f"| {idea.name} | {idea.expected_value} | {columns} | {rationale} |")
        lines.append("")
    return "\n".join(lines) + "\n"
