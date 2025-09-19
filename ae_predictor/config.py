from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Paths:
    raw_csv: str
    processed_csv: str
    train_csv: str
    test_csv: str
    model_dir: str
    reports_dir: str
    figures_dir: str

@dataclass
class Columns:
    numeric: list
    categorical: list
    text: list

@dataclass
class DataCfg:
    target: str
    test_size: float
    random_state: int
    positive_label: int

@dataclass
class XGBCfg:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float

@dataclass
class LogRegCfg:
    C: float
    max_iter: int

@dataclass
class ModelingCfg:
    xgboost: XGBCfg
    logistic_regression: LogRegCfg

@dataclass
class Config:
    paths: Paths
    data: DataCfg
    columns: Columns
    modeling: ModelingCfg
