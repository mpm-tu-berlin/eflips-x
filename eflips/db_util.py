import multiprocessing
import os
import re
import subprocess
import uuid
from configparser import ConfigParser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import gettempdir
from typing import Optional, Union

import alembic.config
import eflips.model
from alembic import command
from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE, INPUTS, TASK_SOURCE
from prefect_sqlalchemy import SqlAlchemyConnector
from sqlalchemy import text


class SimulationStage(Enum):
    """Enum representing different stages of simulation processing."""

    IMPORTED = "imported"
    """A Scenario that has just been imported."""

    CLEANED = "cleaned"
    """Data has been cleaned up after import."""

    BLOCKS_BUILT = "blocks_built"
    """Block building has been run on the Scenario."""

    DEPOTS_ASSIGNED = "depots_assigned"
    """Depot-Block assignments have been made."""

    ENERGY_CONSUMPTION_INITIAL = "energy_consumption_initial"
    """Initial energy consumption has been calculated."""

    DEPOT_LAYOUT_GENERATED = "depot_layout_generated"
    """Depot layout has been generated."""

    SIMULATED = "simulated"
    """The Scenario has been simulated."""

    SMART_CHARGING_COMPUTED = "smart_charging_computed"
    """Smart charging has been computed for the Scenario."""


@dataclass
class DatabaseConfig:
    """Container for database connection information."""

    database: str
    host: str
    port: int
    username: str
    password: Optional[str] = None

    @classmethod
    def from_connector(cls, connector: SqlAlchemyConnector) -> "DatabaseConfig":
        """Create DatabaseConfig from SqlAlchemyConnector."""
        info = connector.connection_info
        return cls(
            database=info.database,
            host=info.host,
            port=info.port,
            username=info.username,
            password=info.password,
        )


class DatabaseUtilities:
    """Utility class for database operations to avoid repetition."""

    def __init__(self):
        self._config = None
        self._pg_paths = None

    @property
    def config(self) -> ConfigParser:
        """Lazy load configuration."""
        if self._config is None:
            self._config = ConfigParser()
            config_path = Path("config.ini")
            if not config_path.exists():
                raise FileNotFoundError("config.ini not found in current directory")
            self._config.read(config_path)
        return self._config

    @property
    def pg_paths(self) -> dict:
        """Get PostgreSQL tool paths from config."""
        if self._pg_paths is None:
            self._pg_paths = {
                "pg_dump": self.config.get("paths", "pg_dump"),
                "pg_restore": self.config.get("paths", "pg_restore"),
            }
            # Validate that paths exist
            for tool, path in self._pg_paths.items():
                if not Path(path).exists():
                    raise FileNotFoundError(f"{tool} not found at {path}")
        return self._pg_paths

    @staticmethod
    def validate_schema_name(schema_name: str) -> None:
        """Validate schema name to prevent SQL injection."""
        if not schema_name:
            raise ValueError("Schema name cannot be empty")

        # Allow only alphanumeric, underscore, and hyphen
        if not re.match(r"^[a-zA-Z0-9_-]+$", schema_name):
            raise ValueError(
                f"Invalid schema name '{schema_name}'. "
                "Only alphanumeric characters, underscores, and hyphens are allowed."
            )

        # Check length
        if len(schema_name) > 63:  # PostgreSQL identifier limit
            raise ValueError(f"Schema name '{schema_name}' is too long (max 63 characters)")

    def run_pg_command(
        self, command: list, db_config: DatabaseConfig, check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a PostgreSQL command with proper error handling."""
        env = os.environ.copy()
        if db_config.password:
            env["PGPASSWORD"] = db_config.password.get_secret_value()

        # Add connection parameters
        command.extend(
            [
                "--host",
                db_config.host,
                "--port",
                str(db_config.port),
                "--username",
                db_config.username,
            ]
        )

        try:
            return subprocess.run(command, check=check, env=env, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"PostgreSQL command failed: {' '.join(command)}\n" f"Error: {e.stderr}"
            ) from e


# Create singleton instance
db_utils = DatabaseUtilities()


@task(name="Clear Database", cache_policy=NO_CACHE)
def clear_database_schema(database_and_schema_id: str) -> None:
    """
    Clear the database schema by dropping the schema.

    :param database_and_schema_id: The schema name and Prefect block identifier
    :raises ValueError: If schema name is invalid
    """
    logger = get_run_logger()
    logger.info(f"Clearing schema: {database_and_schema_id}")

    # Validate schema name
    db_utils.validate_schema_name(database_and_schema_id)

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        # Drop schemas using parameterized queries where possible
        # Note: Schema names can't be parameterized in DDL, but we've validated above
        db_connector.execute(f"DROP SCHEMA IF EXISTS {database_and_schema_id} CASCADE")
        db_connector.execute("DROP SCHEMA IF EXISTS temp CASCADE")
        db_connector.execute("CREATE SCHEMA IF NOT EXISTS common")
        db_connector.execute("CREATE EXTENSION IF NOT EXISTS postgis SCHEMA common;")
        db_connector.execute("CREATE EXTENSION IF NOT EXISTS btree_gist SCHEMA common;")

    logger.info("Schema cleared successfully")


@task(name="Update Database Schema")
def update_database_schema(database_and_schema_id: str) -> None:
    """
    Update the database schema by applying Alembic migrations.

    :param database_and_schema_id: The schema name and Prefect block identifier
    """
    logger = get_run_logger()
    logger.info(f"Updating database schema: {database_and_schema_id}")

    # Validate schema name
    db_utils.validate_schema_name(database_and_schema_id)

    # The path where the model is located is also where the Alembic migrations are stored
    model_dir = Path(eflips.model.__file__).parent

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        db_config = DatabaseConfig.from_connector(db_connector)

        # Temporarily alter the database's search path to include the schema
        db_connector.execute(
            f"ALTER DATABASE {db_config.database} SET search_path TO {database_and_schema_id},common;"
        )

        # Run alembic migrations using subprocess
        config = ConfigParser()
        config.read("config.ini")
        sqlalchemy_url = db_connector.connection_info.create_url().render_as_string()

        result = subprocess.run(
            [
                config.get("paths", "alembic"),
                "upgrade",
                "head",
            ],
            check=True,
            env={
                "DATABASE_URL": sqlalchemy_url,
            },
            cwd=str(model_dir),
        )

        # Reset the search path to include the common schema
        db_connector.execute(
            f"ALTER DATABASE {db_config.database} SET search_path TO public,common;"
        )

    logger.info("Database schema updated successfully")


def get_snapshot_path(
    database_and_schema_id: str, checkpoint_name: Union[str, SimulationStage], snapshot_folder: str
) -> Path:
    """
    Generate the path for the snapshot file.

    :param database_and_schema_id: Schema identifier
    :param checkpoint_name: Checkpoint name or SimulationStage
    :param snapshot_folder: Base folder for snapshots
    :return: Path to the snapshot file
    """
    # Handle SimulationStage enum
    if isinstance(checkpoint_name, SimulationStage):
        checkpoint_name = checkpoint_name.value

    snapshot_dir = Path(snapshot_folder) / database_and_schema_id
    return snapshot_dir / f"{checkpoint_name}.pg_dump"


@task(
    cache_key_fn=lambda context, parameters: f"db_snapshot_{parameters['database_and_schema_id']}"
    f"_{parameters['checkpoint_name']}",
)
def create_db_snapshot(
    database_and_schema_id: str, checkpoint_name: Union[str, SimulationStage], snapshot_folder: str
) -> str:
    """
    Create a snapshot of the database using a safe schema renaming approach.

    :param database_and_schema_id: Schema name and Prefect block identifier
    :param checkpoint_name: Name/stage for the checkpoint
    :param snapshot_folder: Path to snapshot storage folder
    :return: Path to the created snapshot file
    :raises ValueError: If inputs are invalid
    :raises RuntimeError: If snapshot creation fails
    """
    logger = get_run_logger()

    # Validate inputs
    db_utils.validate_schema_name(database_and_schema_id)
    if not snapshot_folder:
        raise ValueError("Snapshot folder path must be provided")

    # Ensure snapshot directory exists
    snapshot_path = get_snapshot_path(database_and_schema_id, checkpoint_name, snapshot_folder)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    # Get database configuration
    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        db_config = DatabaseConfig.from_connector(db_connector)

    # Use a unique temp file to avoid conflicts
    temp_folder = Path(gettempdir()) / f"{uuid.uuid4().hex}.pg_dump"

    try:
        # Step 1: Dump current schema to temp file
        logger.info("Creating temporary snapshot...")
        dump_cmd = [
            db_utils.pg_paths["pg_dump"],
            "-Fd",
            "-d",
            db_config.database,
            "--schema",
            database_and_schema_id,
            "-f",
            str(temp_folder),
            "-j",
            str(multiprocessing.cpu_count()),
        ]
        db_utils.run_pg_command(dump_cmd, db_config)

        # Step 2: Rename schema with transaction safety
        logger.info(f"Renaming schema {database_and_schema_id} to 'temp'...")
        with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
            # Check if temp schema already exists (safety check)
            result = db_connector.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'temp'"
            )
            if result.fetchone():
                raise RuntimeError("Temp schema already exists - possible incomplete previous run")

            db_connector.execute(f"ALTER SCHEMA {database_and_schema_id} RENAME TO temp")

        # Step 3: Restore from temp file to original schema name
        logger.info(f"Restoring snapshot to schema {database_and_schema_id}...")
        restore_cmd = [
            db_utils.pg_paths["pg_restore"],
            "-d",
            db_config.database,
            "-e",
            "-Fd",
            "-j",
            str(multiprocessing.cpu_count()),
            str(temp_folder),
        ]
        db_utils.run_pg_command(restore_cmd, db_config)

        # Step 4: Dump temp schema to final snapshot
        logger.info(f"Creating final snapshot at {snapshot_path}...")
        final_dump_cmd = [
            db_utils.pg_paths["pg_dump"],
            "-Fd",
            "-d",
            db_config.database,
            "--schema",
            "temp",
            "-f",
            str(snapshot_path),
            "-j",
            str(multiprocessing.cpu_count()),
        ]
        db_utils.run_pg_command(final_dump_cmd, db_config)

        # Step 5: Clean up temp schema
        logger.info("Cleaning up temp schema...")
        with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
            db_connector.execute("DROP SCHEMA temp CASCADE")

        logger.info(f"Snapshot created successfully at {snapshot_path}")
        return str(snapshot_path)

    finally:
        # Clean up temp folder (and subfiles)
        if os.path.exists(temp_folder):
            try:
                # Recursively remove temp directory
                for item in temp_folder.glob("*"):
                    if item.is_file():
                        item.unlink()
                os.rmdir(temp_folder)
            except Exception as e:
                logger.error(f"Failed to clean up temporary folder {temp_folder}: {e}")
                # Best effort cleanup, don't raise here


@task()
def restore_db_snapshot(
    database_and_schema_id: str,
    checkpoint_name: Optional[Union[str, SimulationStage]] = None,
    snapshot_path: Optional[str] = None,
    snapshot_folder: Optional[str] = None,
) -> None:
    """
    Restore a database from a snapshot.

    :param database_and_schema_id: Schema name and Prefect block identifier
    :param checkpoint_name: Checkpoint name (if looking up snapshot)
    :param snapshot_path: Direct path to snapshot file
    :param snapshot_folder: Folder containing snapshots (used with checkpoint_name)
    :raises ValueError: If parameters are invalid
    :raises FileNotFoundError: If snapshot file doesn't exist
    """
    logger = get_run_logger()

    # Validate inputs
    db_utils.validate_schema_name(database_and_schema_id)

    # Determine snapshot path
    if checkpoint_name is not None and snapshot_path is None:
        if not snapshot_folder:
            snapshot_folder = gettempdir()
        path_to_restore = get_snapshot_path(
            database_and_schema_id, checkpoint_name, snapshot_folder
        )
        if not path_to_restore.exists():
            raise FileNotFoundError(f"Snapshot file not found: {path_to_restore}")
    elif checkpoint_name is None and snapshot_path is not None:
        path_to_restore = Path(snapshot_path)
        if not path_to_restore.exists():
            raise FileNotFoundError(f"Snapshot file not found: {path_to_restore}")
    else:
        raise ValueError("Either checkpoint_name or snapshot_path must be provided, but not both")

    # Clear existing schema
    clear_database_schema(database_and_schema_id)

    # Get database configuration
    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        db_config = DatabaseConfig.from_connector(db_connector)

    try:
        # Restore to temp schema
        logger.info("Restoring database to 'temp' schema...")
        restore_cmd = [
            db_utils.pg_paths["pg_restore"],
            "-d",
            db_config.database,
            "-e",
            "-Fd",
            "-j",
            str(multiprocessing.cpu_count()),
            str(path_to_restore),
        ]
        db_utils.run_pg_command(restore_cmd, db_config)

        # Rename temp to target schema
        logger.info(f"Renaming 'temp' schema to {database_and_schema_id}...")
        with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
            db_connector.execute(f"ALTER SCHEMA temp RENAME TO {database_and_schema_id}")

        logger.info("Database restored successfully")

    except Exception as e:
        # Try to clean up temp schema on failure
        logger.error(f"Restore failed: {e}")
        try:
            with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
                db_connector.execute("DROP SCHEMA IF EXISTS temp CASCADE")
        except Exception:
            pass  # Best effort cleanup
        raise
