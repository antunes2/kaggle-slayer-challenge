# Questions (`questions/`)

Open questions awaiting a decision. One file per question, named `Q####-slug.md`,
numbered sequentially. A question raised here should **block nothing**: if it blocks
work, say so in the file and provide a working default in the meantime.

## Template

```markdown
# Q#### — <title>

- **Status:** open | settled → D####
- **Raised:** YYYY-MM-DD
- **Blocks:** what this holds up (or "nothing — default in place")

## Question
What exactly needs to be decided.

## Context
Why it matters now; relevant facts and constraints.

## Options
1. ... — tradeoffs
2. ...
3. ...

## Recommendation
What we lean toward and why; when to revisit.
```

## Index

- [Q0001 — Which OpenRouter model(s) should each agent use?](Q0001-openrouter-models-per-agent.md)
- [Q0002 — How should the harness handle slow or pathological candidates?](Q0002-harness-timeout-handling.md)
- [Q0003 — What happens when the CLI is stopped mid-run?](Q0003-interruption-safety.md)
- [Q0004 — How does the user pick a competition?](Q0004-competition-discovery-ux.md)
- [Q0005 — How does the run act on falling short of the target?](Q0005-iterate-reentry-on-shortfall.md) (settled → D0007)
