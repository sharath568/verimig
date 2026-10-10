from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Case:
    id: str
    category: str
    observability: str
    title: str
    source: str
    golden: str
    calls: list[str]
    setup: str = ""
    ordered: bool = True
    differential: bool = True
    compare_messages: bool = False
    review: bool = False
    split: str = "dev"  # "dev" or "heldout" (cases written after the trap checklist was frozen)
    notes: str = ""
    extra: dict = field(default_factory=dict)


_KNOWN = set(Case.__dataclass_fields__)


def load_corpus(path: str | Path) -> list[Case]:
    data = yaml.safe_load(Path(path).read_text())
    cases = []
    for raw in data["cases"]:
        known = {k: v for k, v in raw.items() if k in _KNOWN}
        extra = {k: v for k, v in raw.items() if k not in _KNOWN}
        known["setup"] = known.get("setup") or ""
        cases.append(Case(**known, extra=extra))
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in corpus")
    return cases
