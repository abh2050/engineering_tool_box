"""Persistence layer for saving and retrieving calculation history."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import Column, JSON as SAJSON

from .state import Step


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DB_PATH = DATA_DIR / "history.db"


class RunRecord(SQLModel, table=True):
    """ORM model capturing a single tool execution."""
    
    __table_args__ = {'extend_existing': True}

    id: int | None = Field(default=None, primary_key=True)
    tool_name: str
    user_query: str
    auto_route: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    inputs: Dict[str, Any] = Field(sa_column=Column(SAJSON), default_factory=dict)
    outputs: Dict[str, Any] = Field(sa_column=Column(SAJSON), default_factory=dict)
    steps: List[Dict[str, Any]] = Field(sa_column=Column(SAJSON), default_factory=list)
    units: Dict[str, str] = Field(sa_column=Column(SAJSON), default_factory=dict)
    warnings: List[str] = Field(sa_column=Column(SAJSON), default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "user_query": self.user_query,
            "auto_route": self.auto_route,
            "created_at": self.created_at.isoformat(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "steps": self.steps,
            "units": self.units,
            "warnings": self.warnings,
        }


class HistoryManager:
    """High-level API for queryable run history."""

    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = Path(database_path or DEFAULT_DB_PATH)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.database_path}")
        SQLModel.metadata.create_all(self.engine)

    def log_run(
        self,
        *,
        tool_name: str,
        user_query: str,
        auto_route: bool,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        steps: Iterable[Step | Dict[str, Any]],
        units: Dict[str, str],
        warnings: Iterable[str] | None = None,
    ) -> int:
        """Persist a completed run and return its identifier."""

        warnings_list = list(warnings or [])
        serialized_steps = [step.to_dict() if isinstance(step, Step) else step for step in steps]
        record = RunRecord(
            tool_name=tool_name,
            user_query=user_query,
            auto_route=auto_route,
            inputs=inputs,
            outputs=outputs,
            steps=serialized_steps,
            units=units,
            warnings=warnings_list,
        )
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        logger.debug("Saved run %s for tool %s", record.id, tool_name)
        return int(record.id)

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent runs ordered by newest first."""

        with Session(self.engine) as session:
            statement = select(RunRecord).order_by(RunRecord.created_at.desc()).limit(limit)
            results = session.exec(statement).all()
        return [record.to_dict() for record in results]

    def get_run(self, run_id: int) -> Dict[str, Any] | None:
        with Session(self.engine) as session:
            record = session.get(RunRecord, run_id)
            return record.to_dict() if record else None

    def delete_run(self, run_id: int) -> bool:
        with Session(self.engine) as session:
            record = session.get(RunRecord, run_id)
            if not record:
                return False
            session.delete(record)
            session.commit()
        logger.debug("Deleted run %s", run_id)
        return True

