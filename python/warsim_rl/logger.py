"""Minimal CSV + stdout metrics logger (no tensorboard dependency)."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Optional, TextIO


class CsvLogger:
    def __init__(self, path: str | Path, stream: Optional[TextIO] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = stream if stream is not None else sys.stdout
        self.header: Optional[list] = None
        if self.path.is_file() and self.path.stat().st_size > 0:
            with self.path.open(newline="", encoding="utf-8") as f:
                self.header = next(csv.reader(f), None)

    def log(self, row: dict) -> None:
        new_file = self.header is None
        if new_file:
            self.header = list(row)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(self.header)
            writer.writerow([_fmt(row.get(k, "")) for k in self.header])

    def echo(self, text: str) -> None:
        print(text, file=self.stream, flush=True)


def _fmt(value: object) -> object:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6g}"
    return value
