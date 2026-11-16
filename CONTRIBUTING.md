# Contributing

Thanks for helping. Bug reports, ideas, docs fixes and code are all welcome.

## Set up

You need Python 3.12 or newer. The tests need no network and no data.

```sh
git clone git@github.com:similie/4Shadow-api.git
cd 4Shadow-api
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

To work against real data, ingest a short slice of the newest run and start the API with reload:

```sh
python -m forecast.ingest --max-fhr 24          # about 300 MB
python -m forecast.places build                 # cities500 place index, seconds
uvicorn forecast.api:app --port 5003 --reload   # http://localhost:5003/v2/docs
```

## Checks

| Command | What it does |
| --- | --- |
| `ruff check .` | Lint: errors and likely bugs (rules in `ruff.toml`) |
| `pytest` | Unit and API tests on a small synthetic store |
| `docker build -t 4shadow-forecast .` | The image CI builds and starts |
| `API=http://127.0.0.1:5003 PYTHON=python deploy/deploy.sh check` | The deploy smoke tests, against a running API with real data |

CI runs lint, the tests on Python 3.12, 3.13 and 3.14, and the Docker image on every pull request. On `main`, daily and by hand, it also ingests the first hours of the newest real GFS run and runs the smoke tests on it, which catches changes in NOAA's files.

## Making a change

- **Open an issue first** for anything bigger than a fix, so we can agree on the approach before you spend time on it.
- **Keep the API stateless and the store plain.** No database or cache that has to be warmed; anything can read the `.npy` files.
- **Don't break clients.** Add fields rather than renaming them. If an answer has to change, say so in `CHANGELOG.md`.
- **Add a test** for new behaviour. `tests/forecast/conftest.py` builds a coarse synthetic run that most tests use.
- **Adding a variable** is mostly one entry in `forecast/catalog.py`; [the guide](forecast/README.md#adding-a-variable) explains the fields.
- **Write for people.** Comments, docs and error messages in plain words, saying what something does and why.
- **Update `CHANGELOG.md`** under "Unreleased" when users would notice the change.
- **Never commit a key or an `.env` file.**

A pull request needs CI green and one maintainer's review.

## Deploying

Maintainers deploy `main` to api.4shadow.io with [`deploy/deploy.sh update`](docs/deploy.md#update). Nothing is published to PyPI or a container registry.

## License

By contributing, you agree that your contribution is released under the [GNU General Public License v3.0](LICENSE).
