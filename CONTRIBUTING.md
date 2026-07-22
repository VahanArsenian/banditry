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

## Release checklist

1. Bump the version in **four** places: `pyproject.toml`, `src/banditry/__init__.py`
   (`__version__`), `CITATION.cff` (`version:`), and the README BibTeX (`version = {...}`).
2. Add a `CHANGELOG.md` entry (and its compare link at the bottom).
3. Run the full gate: `pytest`, `ruff check .`, `ruff format --check .`,
   `python -m build && twine check dist/*`.
4. Commit, push `main`, wait for CI, then push the `vX.Y.Z` tag (triggers the
   PyPI publish) and create the GitHub Release with the changelog notes
   (triggers the Zenodo archive).
5. **After Zenodo archives the release** (a minute or two), update the
   *version DOI* in `CITATION.cff` (`doi:`) and the README BibTeX (`doi = {...}`)
   — Zenodo mints a **new version DOI for every release**. The concept DOI in
   `CITATION.cff` `identifiers:` and the README badge never change. Get the new
   DOI from `curl -sI https://zenodo.org/badge/latestdoi/<repo-id>`.

## Guidelines

- Keep pull requests focused — one change per PR.
- Add or update tests for behaviour changes. Agent-level tests should assert
  plumbing invariants (shapes, bounds, bookkeeping) with tiny, seeded budgets —
  never solution quality.
- Public API changes (renames, removals) need a deprecation path: keep the old
  name working with a `DeprecationWarning` for at least one minor release.
- Update `CHANGELOG.md` under an `Unreleased`/next-version heading.
