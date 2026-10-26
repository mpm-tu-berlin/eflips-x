# eflips-x

Prefect-based orchestrator for electric bus network simulations. Chains eFLIPS components (model, depot, ingest, eval,
opt, tco) into cacheable pipelines. Python 3.12/3.13, Poetry.

## Architecture

Steps live in `eflips/x/steps/{generators,modifiers,analyzers}`; base classes in `eflips/x/framework/__init__.py`.
Each step writes `step_NNN_ClassName.db` to `work_dir`; the next step copies it.

- `Generator`: creates a DB from input files. Cache key ignores previous DB.
- `Modifier`: copies and modifies the previous DB. Cache key includes previous DB hash.
- `Analyzer`: read-only (runs on a temp copy), returns results from `analyze()`, optional `visualize()`.
- Parameters come from `PipelineContext.params`: global keys unprefixed, step keys prefixed with the class name
  (`"VehicleScheduling.charge_type"`). Case-sensitive. Every step implements `document_params()`.
- `"scenario_display_config"` (a `ScenarioDisplayConfig`) controls scenario order/names/baseline in multi-scenario plots.
- Every step constructor takes `code_version`; bump it whenever the step's logic changes (cache invalidation).
- Always use `with context.get_session() as session:`.
- Deleting a step's output DB from `work_dir` forces that step (and downstream steps) to re-run.
- Failed Modifier DBs are renamed to `.{timestamp}.failed`.

## Commands

```bash
export SPATIALITE_LIBRARY_PATH=/opt/homebrew/lib/mod_spatialite.dylib   # Linux: /usr/lib/x86_64-linux-gnu/mod_spatialite.so
export PYTHONPATH=tests:.
poetry run pytest -n auto                          # one test: path::test_name
poetry run black .                                 # line length 99
poetry run mypy eflips --explicit-package-bases --strict
cd doc && poetry run sphinx-build -b html . _build
```

Inspect `.db` files with `spatialite`, not `sqlite3`. Routing needs `OPENROUTESERVICE_BASE_URL` (and
`OPENROUTESERVICE_API_KEY` for the official server).

## Rules

- Code must pass `black --check` and `mypy --strict` (enforced by the Stop hook and CI).
- Matplotlib: use `plt.subplots(layout="constrained")`, never `tight_layout`.
- Never edit `poetry.lock` by hand; change dependencies via `poetry add`. Ask before adding a runtime dependency.
- Do not edit or delete existing tests to make them pass; report tests that look wrong.

## Pull requests

Every PR bumps the version (SemVer): patch = no API change, minor = backwards-compatible API change, major = breaking.
Version lives in `pyproject.toml` (`version`) and `doc/conf.py` (`release`); CI fails if they differ.
Branches: `feature/...`, `fix/...`, merged via PR into `main`.
