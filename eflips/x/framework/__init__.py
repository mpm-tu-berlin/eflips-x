"""
Re-Useable eflips-x framework components. Uses abastract base classes in order to create prototypes for various kinds of
eflips-x modules.
"""

import hashlib
import shutil
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import eflips.model
import sqlalchemy.orm.session
from eflips.model import Base
from prefect import task
from prefect.artifacts import create_markdown_artifact
from sqlalchemy.exc import PendingRollbackError
from sqlalchemy.orm import Session


@dataclass
class PipelineContext:
    """Manages pipeline execution state and database chaining."""

    work_dir: Path
    params: Dict[str, Any] = field(default_factory=dict)
    current_db: Optional[Path] = None
    step_count: int = 0
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def get_next_db_path(self, step_name: str) -> Path:
        """Generate path for next database state."""
        self.step_count += 1
        return self.work_dir / f"step_{self.step_count:03d}_{step_name}.db"

    def set_current_db(self, db_path: Path):
        """Update current database reference."""
        self.current_db = db_path
        return db_path


class PipelineStep(ABC):
    """Base class for all pipeline steps."""

    def __init__(self, code_version: str, cache_enabled: bool = True, **kwargs):
        if not code_version:
            raise ValueError(
                "code_version must be provided for cache invalidation. Please set a default value "
                "(say 'v1') in the subclass method signature."
            )
        self.code_version = code_version
        self.cache_enabled = cache_enabled
        self.config = kwargs
        self._prefect_task = None

    @abstractmethod
    def document_params(self) -> Dict[str, str]:
        """
        This method documents the parameters of the generator. It returns a dictionary where the keys are the parameter
        and the values are a description of the parameter. The values may use markdown formatting. They may be
        multi-line strings.
        If the parameters are specific to a subclass, the key should be prefixed with the class name and a dot.
        For example, if the class is MyGenerator and the parameter is my_param, the key should be MyGenerator.my_param.
        :return: A dictionary documenting the parameters of the generator.
        """
        pass

    @abstractmethod
    def compute_cache_key(self, context: PipelineContext) -> str:
        """Compute cache key for this step."""
        pass

    @abstractmethod
    def execute_impl(self, context: PipelineContext) -> None:
        """Execute the step logic. To be implemented by subclasses."""
        pass

    def execute(self, context: PipelineContext) -> None:
        """Execute the step with Prefect task wrapping."""
        if self._prefect_task is None:
            self._create_prefect_task()

        return self._prefect_task(context)

    def _create_prefect_task(self):
        """Create a Prefect task for this step."""

        @task(
            name=self.__class__.__name__,
            cache_key_fn=lambda ctx, parameters: (
                self.compute_cache_key(parameters["context"]) if self.cache_enabled else None
            ),
        )
        def task_wrapper(context: PipelineContext):
            result = self.execute_impl(context)

            # Create artifact for observability
            create_markdown_artifact(
                key=f"{self.__class__.__name__.lower()}-completion",
                markdown=self._create_artifact_markdown(context, result),
                description=f"Step {self.__class__.__name__} completed",
            )

            return result

        self._prefect_task = task_wrapper

    @abstractmethod
    def _create_artifact_markdown(self, context: PipelineContext, result: Any) -> str:
        """Create markdown for artifact logging. To be implemented by subclasses."""
        pass

    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of a file."""
        if not filepath.exists():
            return "missing"

        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _compute_poetry_lock_hash(self) -> str:
        """Hash poetry.lock for dependency tracking."""
        poetry_lock = Path("poetry.lock")
        if poetry_lock.exists():
            return self._compute_file_hash(poetry_lock)
        else:
            warnings.warn("poetry.lock not found")
            return "missing"


class Generator(PipelineStep):
    """
    Generator creates a new database from input files and parameters.
    Does NOT depend on previous database state for cache invalidation.
    """

    def __init__(
        self,
        input_files: Optional[List[Union[str, Path]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_files = [Path(f) for f in (input_files or [])]

    def compute_cache_key(self, context: PipelineContext) -> str:
        """Cache key based on input files and parameters, NOT database (since it doesn't exist yet)."""
        key_parts = [
            f"generator:{self.__class__.__name__}",
            f"code:{self.code_version}",
            f"deps:{self._compute_poetry_lock_hash()}",
        ]

        # Hash input files
        for filepath in self.input_files:
            key_parts.append(f"file:{filepath.name}:{self._compute_file_hash(filepath)}")

        # Hash relevant parameters
        if context.params:
            params_str = str(sorted(context.params.items()))
            params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
            key_parts.append(f"params:{params_hash}")

        return ":".join(key_parts)

    def execute_impl(self, context: PipelineContext) -> Path:
        """Execute generator: create new database."""
        output_db = context.get_next_db_path(self.__class__.__name__)

        # Give a clear error if the working directory does not exist
        if not context.work_dir.exists():
            raise FileNotFoundError(f"Working directory {context.work_dir} does not exist.")

        # Set up the new database and open a session
        if output_db.exists():
            raise FileExistsError(f"Output database {output_db} already exists.")
        db_url = f"sqlite:////{output_db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        Base.metadata.create_all(engine)
        session = Session(engine)

        try:
            result = self.generate(session, context.params)
            session.commit()
            session.close()
        except Exception as e:
            # Try to commit anyway
            try:
                session.commit()
            except PendingRollbackError:
                pass
            session.close()
            # If generation fails, move the incomplete DB file to a .failed extension
            now = datetime.now().isoformat()
            failed_db = output_db.with_suffix(f"-{now}.failed")
            shutil.move(output_db, failed_db)
            raise e
        finally:
            engine.dispose()
            context.set_current_db(output_db)
        return result

    @abstractmethod
    def generate(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> Path:
        """Generate new database. To be implemented by subclasses."""
        pass

    def _create_artifact_markdown(self, context: PipelineContext, result: Path) -> str:
        return f"""## {self.__class__.__name__} Completed
        
Type: Generator
Output Database: `{result}`
Input Files: {', '.join(str(f) for f in self.input_files)}"""


class Modifier(PipelineStep):
    """
    Modifier transforms an existing database to produce a new one.
    Depends on previous database state for cache invalidation.
    """

    def __init__(self, additional_files: Optional[List[Union[str, Path]]] = None, **kwargs):
        super().__init__(**kwargs)
        self.additional_files = [Path(f) for f in (additional_files or [])]

    def compute_cache_key(self, context: PipelineContext) -> str:
        """Cache key based on input database, additional files, and parameters."""
        key_parts = [
            f"modifier:{self.__class__.__name__}",
            f"code:{self.code_version}",
            f"deps:{self._compute_poetry_lock_hash()}",
        ]

        # Hash input database
        if context.current_db:
            key_parts.append(f"db:{self._compute_file_hash(context.current_db)}")

        # Hash additional files
        for filepath in self.additional_files:
            key_parts.append(f"file:{filepath.name}:{self._compute_file_hash(filepath)}")

        # Hash relevant parameters
        if context.params:
            params_str = str(sorted(context.params.items()))
            params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
            key_parts.append(f"params:{params_hash}")

        return ":".join(key_parts)

    def execute_impl(self, context: PipelineContext) -> Path:
        """Execute modifier: copy database and modify it."""
        if not context.current_db:
            raise ValueError(f"Modifier {self.__class__.__name__} requires an input database")

        output_db = context.get_next_db_path(self.__class__.__name__)

        # Set up the new database and open a session
        if output_db.exists():
            raise FileExistsError(f"Output database {output_db} already exists.")
        # Copy input to output to preserve intermediate states
        shutil.copy2(context.current_db, output_db)

        db_url = f"sqlite:////{output_db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            result = self.modify(session, context.params)
            session.commit()
            session.close()
        except Exception as e:
            # Try to commit anyway
            try:
                session.commit()
            except PendingRollbackError:
                pass
            session.close()
            # If modification fails, move the incomplete DB file to a .failed extension
            now = datetime.now().isoformat()
            failed_db = output_db.with_suffix(f"-{now}.failed")
            shutil.move(output_db, failed_db)
            raise e
        finally:
            engine.dispose()
            context.set_current_db(output_db)
        return result

    @abstractmethod
    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> Path:
        """Modify database in place. To be implemented by subclasses."""
        pass

    def _create_artifact_markdown(self, context: PipelineContext, result: Path) -> str:
        return f"""## {self.__class__.__name__} Completed
        
Type: Modifier
Input Database: `{context.current_db}`
Output Database: `{result}`
Additional Files: {', '.join(str(f) for f in self.additional_files)}"""


class Analyzer(PipelineStep):
    """
    Analyzer reads a database and produces reports/artifacts.
    Does NOT modify the database.
    """

    def __init__(self, additional_files: Optional[List[Union[str, Path]]] = None, **kwargs):
        super().__init__(**kwargs)
        self.additional_files = [Path(f) for f in (additional_files or [])]

    def compute_cache_key(self, context: PipelineContext) -> str:
        """Cache key based on input database, additional files, and parameters."""
        key_parts = [
            f"analyzer:{self.__class__.__name__}",
            f"code:{self.code_version}",
            f"deps:{self._compute_poetry_lock_hash()}",
        ]

        # Hash input database
        if context.current_db:
            key_parts.append(f"db:{self._compute_file_hash(context.current_db)}")

        # Hash additional files
        for filepath in self.additional_files:
            key_parts.append(f"file:{filepath.name}:{self._compute_file_hash(filepath)}")

        # Hash relevant parameters
        if context.params:
            params_str = str(sorted(context.params.items()))
            params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
            key_parts.append(f"params:{params_hash}")

        return ":".join(key_parts)

    def execute_impl(self, context: PipelineContext) -> Any:
        """Execute analyzer: analyze database without modification."""
        if not context.current_db:
            raise ValueError(f"Analyzer {self.__class__.__name__} requires an input database")

        db_url = f"sqlite:////{context.current_db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            result = self.analyze(context.current_db, context.params)
        finally:
            # Explicitly rollback any uncommitted transactions
            session.rollback()
            session.close()
            engine.dispose()

        # Store result in context artifacts
        context.artifacts[self.__class__.__name__] = result

        return result

    @abstractmethod
    def analyze(self, db: Path, params: Dict[str, Any]) -> Any:
        """Analyze database and return results. To be implemented by subclasses."""
        pass

    def _create_artifact_markdown(self, context: PipelineContext, result: Any) -> str:
        return f"""## {self.__class__.__name__} Completed

Type: Analyzer  
Input Database: `{context.current_db}`
Result Type: {type(result).__name__}"""
