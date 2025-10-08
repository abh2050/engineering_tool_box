"""Search indexing across tools, glossary entries, and run history."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import json

import yaml
from rapidfuzz import fuzz

from .history import HistoryManager
from .routing import ToolMetadata, load_tool_metadata


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(slots=True)
class SearchDocument:
    identifier: str
    kind: str
    title: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    identifier: str
    kind: str
    title: str
    snippet: str
    score: float
    metadata: Dict[str, str]


class SearchIndex:
    """Composite search index with lightweight keyword scoring."""

    def __init__(
        self,
        *,
        tool_registry: Iterable[ToolMetadata] | None = None,
        history_manager: HistoryManager | None = None,
    ) -> None:
        self.tool_registry = list(tool_registry or load_tool_metadata())
        self.history_manager = history_manager or HistoryManager()
        self.glossary_entries = self._load_glossary(DATA_DIR / "glossary.yaml")

    def _load_glossary(self, path: Path) -> List[SearchDocument]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or []
        documents: List[SearchDocument] = []
        for entry in data:
            documents.append(
                SearchDocument(
                    identifier=f"glossary:{entry['id']}",
                    kind="formula",
                    title=entry["name"],
                    content=entry.get("description", ""),
                    metadata={
                        "symbols": ", ".join(entry.get("symbols", [])),
                        "equation": entry.get("equation", ""),
                    },
                )
            )
        return documents

    def refresh_glossary(self) -> None:
        self.glossary_entries = self._load_glossary(DATA_DIR / "glossary.yaml")

    def _tool_documents(self) -> List[SearchDocument]:
        return [
            SearchDocument(
                identifier=f"tool:{tool.name}",
                kind="tool",
                title=tool.name,
                content=" ".join([tool.description, " ".join(tool.keywords)]),
                metadata={"description": tool.description},
            )
            for tool in self.tool_registry
        ]

    def _history_documents(self) -> List[SearchDocument]:
        documents: List[SearchDocument] = []
        for record in self.history_manager.list_runs(limit=200):
            title = f"{record['tool_name']} ({record['created_at']})"
            content_parts = [record.get("user_query", "")] + [json.dumps(record.get("inputs", {}))]
            documents.append(
                SearchDocument(
                    identifier=f"history:{record['id']}",
                    kind="history",
                    title=title,
                    content=" \n".join(content_parts),
                    metadata={
                        "tool": record["tool_name"],
                        "created_at": record["created_at"],
                    },
                )
            )
        return documents

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        documents = self._tool_documents() + self.glossary_entries + self._history_documents()
        scored: List[SearchResult] = []
        for document in documents:
            score = fuzz.WRatio(query, f"{document.title} {document.content}")
            if score < 40:
                continue
            snippet = document.content[:160].strip()
            if len(document.content) > 160:
                snippet += "…"
            scored.append(
                SearchResult(
                    identifier=document.identifier,
                    kind=document.kind,
                    title=document.title,
                    snippet=snippet,
                    score=float(score),
                    metadata=document.metadata,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]


_default_index: SearchIndex | None = None


def _get_default_index() -> SearchIndex:
    global _default_index
    if _default_index is None:
        _default_index = SearchIndex()
    return _default_index


def search(query: str, limit: int = 10) -> List[SearchResult]:
    """Module-level convenience wrapper."""

    return _get_default_index().search(query, limit=limit)

