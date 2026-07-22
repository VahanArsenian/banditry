# Contributing to banditry

Thanks for your interest in contributing! Issues and pull requests are welcome.

## Development setup

Requires Python 3.10+.

```bash
git clone https://github.com/VahanArsenian/banditry.git
cd banditry
pip install -e ".[dev,nuts]"
```

## Running the checks

CI runs the same three commands on every pull request:

```bash
ruff check .          # lint
ruff format --check . # formatting
pytest -m "not slow"  # tests (the full suite, including slow NUTS tests: pytest)
```

Please make sure they pass locally before opening a PR.

## Guidelines

- Keep pull requests focused — one change per PR.
- Add or update tests for behaviour changes. Agent-level tests should assert
  plumbing invariants (shapes, bounds, bookkeeping) with tiny, seeded budgets —
  never solution quality.
- Public API changes (renames, removals) need a deprecation path: keep the old
  name working with a `DeprecationWarning` for at least one minor release.
- Update `CHANGELOG.md` under an `Unreleased`/next-version heading.
