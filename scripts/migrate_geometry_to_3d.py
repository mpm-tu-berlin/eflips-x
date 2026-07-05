#!/usr/bin/env python3
"""In-place migration: upgrade 2D (XY) SpatiaLite geometry columns to 3D (XYZ).

Background
----------
Older eflips-model versions created geometry columns as 2D (``POINT``,
``LINESTRING``, ``POLYGON``).  eflips-model >= 11.2 declares them as 3D
(``POINTZ``/``LINESTRINGZ``/``POLYGONZ``), and code paths such as
``DepotRotationOptimizer.write_optimization_results`` (via
``eflips.model.util.geometry_has_z``) now build 3D geometries.  Inserting a 3D
geometry into a 2D-constrained column fails with::

    Station.geom violates Geometry constraint [geom-type or SRID not allowed]

This script converts the cached databases in place so their geometry columns
match the current (3D) model, avoiding a full re-ingest.  Existing geometries
are lifted to XYZ with ``Z = 0`` (they were 2D, so no Z information was ever
stored); anything that later needs a real altitude fetches it separately.

For each 2D geometry column it:

1. disables + drops the spatial index,
2. de-registers the column (``DiscardGeometryColumn``),
3. rewrites every value with ``CastToXYZ`` (adds ``Z = 0``),
4. re-registers the column as ``XYZ`` (``RecoverGeometryColumn``),
5. rebuilds the spatial index.

Usage
-----
    poetry run python scripts/migrate_geometry_to_3d.py [DB ...]

With no arguments it discovers ``data/cache/**/*.db`` under the repo root
(skipping SpatiaLite journals and ``*.failed`` snapshots).  A ``.bak`` copy of
each database is made first unless ``--no-backup`` is given.  Re-running is
safe: columns already 3D are skipped.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

# geometry_type code (2D) -> canonical type name understood by RecoverGeometryColumn
_TYPE_NAMES = {
    1: "POINT",
    2: "LINESTRING",
    3: "POLYGON",
    4: "MULTIPOINT",
    5: "MULTILINESTRING",
    6: "MULTIPOLYGON",
    7: "GEOMETRYCOLLECTION",
}


def _load_spatialite(con: sqlite3.Connection) -> None:
    """Load mod_spatialite, honouring SPATIALITE_LIBRARY_PATH as a fallback."""
    con.enable_load_extension(True)
    candidates = ["mod_spatialite"]
    env_path = os.environ.get("SPATIALITE_LIBRARY_PATH")
    if env_path:
        # load_extension wants the path without the file-extension suffix.
        candidates.insert(0, env_path.rsplit(".", 1)[0])
    last_err: Exception | None = None
    for cand in candidates:
        try:
            con.load_extension(cand)
            return
        except sqlite3.OperationalError as exc:  # pragma: no cover - env dependent
            last_err = exc
    raise RuntimeError(f"Could not load mod_spatialite (tried {candidates}): {last_err}")


def _real_table_name(con: sqlite3.Connection, lower_name: str) -> str:
    """geometry_columns stores lower-case names; return the real table casing."""
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=?",
        (lower_name,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Table {lower_name!r} referenced in geometry_columns is missing")
    return str(row[0])


def _columns_to_migrate(con: sqlite3.Connection) -> List[Tuple[str, str, int, int, int]]:
    """Return (table, column, geometry_type, srid, spatial_index_enabled) for 2D columns."""
    rows = con.execute(
        "SELECT f_table_name, f_geometry_column, geometry_type, coord_dimension, "
        "srid, spatial_index_enabled FROM geometry_columns"
    ).fetchall()
    result = []
    for table, column, gtype, coord_dim, srid, sidx in rows:
        # coord_dimension is 2 / 3 (integer) or 'XY' / 'XYZ' (text) depending on mode.
        is_2d = str(coord_dim).upper() in ("2", "XY")
        if is_2d:
            result.append((table, column, int(gtype), int(srid), int(sidx or 0)))
    return result


def migrate_db(path: Path, make_backup: bool) -> bool:
    """Migrate one database. Returns True if any column was changed."""
    con = sqlite3.connect(str(path))
    try:
        _load_spatialite(con)
        # Sanity: is this even a spatial DB?
        has_geom_meta = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='geometry_columns'"
        ).fetchone()
        if has_geom_meta is None:
            print(f"  {path}: no geometry_columns table — skipped")
            return False

        todo = _columns_to_migrate(con)
        if not todo:
            print(f"  {path}: all geometry columns already 3D — skipped")
            return False
    finally:
        con.close()

    if make_backup:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"  {path}: backup -> {backup.name}")

    con = sqlite3.connect(str(path))
    con.isolation_level = None  # manual transaction control
    try:
        _load_spatialite(con)
        con.execute("BEGIN")
        for table_lc, column, gtype, srid, sidx in todo:
            table = _real_table_name(con, table_lc)
            type_name = _TYPE_NAMES.get(gtype)
            if type_name is None:
                raise RuntimeError(f"Unsupported geometry_type {gtype} on {table}.{column}")

            if sidx:
                con.execute("SELECT DisableSpatialIndex(?, ?)", (table, column))
                con.execute(f'DROP TABLE IF EXISTS "idx_{table}_{column}"')

            if con.execute("SELECT DiscardGeometryColumn(?, ?)", (table, column)).fetchone()[
                0
            ] != 1:
                raise RuntimeError(f"DiscardGeometryColumn failed for {table}.{column}")

            con.execute(
                f'UPDATE "{table}" SET "{column}" = CastToXYZ("{column}") '
                f'WHERE "{column}" IS NOT NULL'
            )

            recovered = con.execute(
                "SELECT RecoverGeometryColumn(?, ?, ?, ?, 'XYZ')",
                (table, column, srid, type_name),
            ).fetchone()[0]
            if recovered != 1:
                raise RuntimeError(
                    f"RecoverGeometryColumn failed for {table}.{column} "
                    f"({type_name}, srid={srid}, XYZ)"
                )

            if sidx:
                created = con.execute(
                    "SELECT CreateSpatialIndex(?, ?)", (table, column)
                ).fetchone()[0]
                if created != 1:
                    raise RuntimeError(f"CreateSpatialIndex failed for {table}.{column}")

            print(f"  {path}: {table}.{column} {type_name} XY -> XYZ")
        con.execute("COMMIT")
        return True
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def discover_dbs(root: Path) -> List[Path]:
    """Find candidate cache DBs, skipping journals and failed snapshots."""
    dbs = []
    for p in sorted((root / "data" / "cache").rglob("*.db")):
        name = p.name
        if name.endswith("-journal") or ".failed" in name:
            continue
        dbs.append(p)
    return dbs


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dbs", nargs="*", type=Path, help="Database files to migrate")
    parser.add_argument(
        "--no-backup", action="store_true", help="Do not create a .bak copy before migrating"
    )
    args = parser.parse_args(argv)

    if args.dbs:
        targets = args.dbs
    else:
        repo_root = Path(__file__).resolve().parent.parent
        targets = discover_dbs(repo_root)
        if not targets:
            print(f"No databases found under {repo_root / 'data' / 'cache'}")
            return 0
        print(f"Discovered {len(targets)} database(s) under data/cache/")

    changed = 0
    failed = 0
    for db in targets:
        if not db.exists():
            print(f"  {db}: does not exist — skipped")
            continue
        print(f"Migrating {db} ...")
        try:
            if migrate_db(db, make_backup=not args.no_backup):
                changed += 1
        except Exception as exc:  # keep going across the batch
            failed += 1
            print(f"  {db}: FAILED — {exc}", file=sys.stderr)

    print(f"\nDone. {changed} database(s) migrated, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
