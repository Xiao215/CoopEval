#!/usr/bin/env python3
"""
Shared loader/aggregator for taxonomy share datasets.

Wraps the canonical `taxonomy_by_game_mechanism_model_player.share_pct.csv`
export so plotting scripts can obtain consistent count/frequency matrices
without re-implementing pivot logic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ALLOWED_DIMS = ("game", "mechanism", "model", "player")
__all__ = ["ALLOWED_DIMS", "TaxonomyDataset"]


def _validate_dims(dims: Sequence[str]) -> list[str]:
    """Ensure requested group dimensions exist in the dataset."""
    cleaned = []
    for dim in dims:
        if dim not in ALLOWED_DIMS:
            raise ValueError(
                f"Unsupported dimension '{dim}'. Allowed: {ALLOWED_DIMS}."
            )
        cleaned.append(dim)
    if not cleaned:
        raise ValueError("At least one grouping dimension is required.")
    return cleaned


@dataclass(slots=True)
class TaxonomyDataset:
    """Canonical taxonomy dataset (per game/mechanism/model/player)."""

    df: pd.DataFrame

    @classmethod
    def from_share_csv(cls, path: Path) -> "TaxonomyDataset":
        """Load a taxonomy share CSV and derive label counts."""
        df = pd.read_csv(path)
        required = {
            "game",
            "mechanism",
            "model",
            "player",
            "label",
            "share_pct",
            "group_count",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns {sorted(missing)} in {path}"
            )
        df["share_pct"] = df["share_pct"].astype(float)
        df["group_count"] = df["group_count"].astype(int)
        df["label_count"] = df["share_pct"] * df["group_count"] / 100.0
        return cls(df)

    def filter(
        self,
        *,
        games: Iterable[str] | None = None,
        mechanisms: Iterable[str] | None = None,
        models: Iterable[str] | None = None,
        players: Iterable[str] | None = None,
    ) -> "TaxonomyDataset":
        """Return a filtered view limited to specified dimension members."""
        df = self.df
        if games is not None:
            df = df[df["game"].isin(list(games))]
        if mechanisms is not None:
            df = df[df["mechanism"].isin(list(mechanisms))]
        if models is not None:
            df = df[df["model"].isin(list(models))]
        if players is not None:
            df = df[df["player"].isin(list(players))]
        return TaxonomyDataset(df.copy())

    def aggregate(self, dimensions: Sequence[str]) -> pd.DataFrame:
        """Aggregate counts/share across arbitrary dimensions."""
        dims = _validate_dims(dimensions)
        group_cols = [*dims, "label"]
        grouped = (
            self.df.groupby(group_cols, dropna=False)
            .agg(
                label_count=("label_count", "sum"),
                group_count=("group_count", "sum"),
            )
            .reset_index()
        )
        grouped["share_pct"] = np.where(
            grouped["group_count"] > 0,
            grouped["label_count"] * 100.0 / grouped["group_count"],
            0.0,
        )
        return grouped

    def matrix(
        self,
        dimension: str,
        *,
        labels: Sequence[str] | None = None,
        groups: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return a pivoted matrix (labels × dimension) for plotting."""
        grouped = self.aggregate([dimension])
        if groups is not None:
            grouped = grouped[grouped[dimension].isin(groups)]
        matrix = grouped.pivot(
            index="label", columns=dimension, values="share_pct"
        ).fillna(0.0)
        if labels is not None:
            matrix = matrix.reindex(index=list(labels), fill_value=0.0)
        else:
            matrix = matrix.sort_index()
        return matrix

    def top_labels(self, k: int | None = None) -> list[str]:
        """Return labels ordered by aggregated count share."""
        grouped = self.aggregate(["mechanism"])
        totals = (
            grouped.groupby("label")["label_count"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        if k is not None and k > 0:
            return totals[:k]
        return totals

    def union_top_labels(
        self,
        dimension: str,
        top_n: int,
    ) -> list[str]:
        """Return ordered labels formed by the union of each group's top-N labels."""
        if top_n <= 0:
            return self.top_labels()
        grouped = self.aggregate([dimension])
        if grouped.empty:
            return []
        union_labels: set[str] = set()
        for group_value in grouped[dimension].unique().tolist():
            subset = grouped[grouped[dimension] == group_value].sort_values(
                "share_pct", ascending=False
            )
            labels = subset["label"].head(top_n)
            union_labels.update(labels)

        global_order = (
            grouped.groupby("label")["label_count"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        return [label for label in global_order if label in union_labels]
