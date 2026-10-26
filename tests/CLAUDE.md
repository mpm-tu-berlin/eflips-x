# Tests

- Fixtures in `conftest.py`: `temp_db`, `db_session`, `test_data_dir` (`data/input/Berlin Testing`),
  `gtfs_test_data_dir` (`data/input/GTFS`).
- Pattern: `PipelineContext(work_dir=temp_db.parent, params={})`, `step.execute(context=context)`, then assert via
  `context.get_session()`.
- CI splits tests with pytest-split; refresh `.test_durations` with `--store-durations` when adding slow tests.
- Requires `SPATIALITE_LIBRARY_PATH` and `PYTHONPATH=tests:.`.
