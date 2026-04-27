"""Optional ML model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ModelDependencyError(RuntimeError):
    pass


def require_module(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:
        raise ModelDependencyError(f"install zeroalpha[ml] or optional package {name}") from exc


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    params: dict[str, Any]


def default_champion_specs() -> list[ModelSpec]:
    return [
        ModelSpec("logistic", {"C": 0.5, "class_weight": "balanced"}),
        ModelSpec("lightgbm", {"objective": "binary", "learning_rate": 0.03, "n_estimators": 150}),
        ModelSpec("catboost", {"loss_function": "Logloss", "iterations": 150, "learning_rate": 0.03}),
        ModelSpec("xgboost", {"objective": "binary:logistic", "learning_rate": 0.03, "n_estimators": 150}),
        ModelSpec("tabicl", {}),
        ModelSpec("tabpfn", {}),
    ]


def make_lightgbm_classifier(params: dict[str, Any]) -> Any:
    lightgbm = require_module("lightgbm")
    return lightgbm.LGBMClassifier(**params)


def make_xgboost_classifier(params: dict[str, Any]) -> Any:
    xgboost = require_module("xgboost")
    return xgboost.XGBClassifier(**params)


def make_catboost_classifier(params: dict[str, Any]) -> Any:
    catboost = require_module("catboost")
    return catboost.CatBoostClassifier(**params)
