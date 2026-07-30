# Contributing

Thank you for improving DrugEvolve.

## Development setup

```bash
python -m pip install -e '.[agents,dev]'
python -m pytest
ruff check drugevolve tests
```

## Design rules

1. Keep `drugevolve/` dependency-light. Agent SDK and cluster integrations
   belong behind optional extras or adapters.
2. Preserve the learn → design → experiment → analyze loop.
3. Keep experiment history and external cognition separate.
4. Every evaluator must have an explicit timeout and structured results.
5. Never broaden mutation scope implicitly.
6. Record failed experiments instead of deleting them from history.
7. Add tests for storage migrations, samplers, path boundaries, and timeout
   behavior.

## Pull requests

- Keep changes focused.
- Explain compatibility impact and migration steps.
- Include a deterministic local test or example.
- Do not include datasets, model checkpoints, API keys, internal endpoints, or
  generated run directories.

## Commit hygiene

Use descriptive commits. Generated experiment artifacts should remain under
`.drugevolve/`, which is ignored by Git.
