## What this changes

<!-- What and why, in a few sentences. Link the issue if there is one. -->

## How it was checked

- [ ] `ruff check .` and `pytest` pass
- [ ] New behaviour has a test, or the PR says why not
- [ ] For ingest or store changes: ingested a real run (`python -m forecast.ingest --max-fhr 6`) and checked the API on it
- [ ] For changes to API answers: existing clients keep working, or `CHANGELOG.md` says what breaks
- [ ] Docs and `CHANGELOG.md` updated where users would notice

<!-- Never include an API key or an .env file. -->
