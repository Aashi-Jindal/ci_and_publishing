from typing import Dict, List
from pydantic import BaseModel
from strictyaml import load

from pathlib import Path

import regression_model

# Project Directories
PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    variables_to_rename: Dict[str, str]
    features: List[str]
    test_size: float
    random_state: int
    alpha: float
    # categorical_vars_with_na_frequent: List[str]
    categorical_vars_with_na_missing: List[str]
    numerical_vars_with_na: List[str]
    temporal_vars: List[str]
    ref_var: str
    numericals_log_vars: List[str]
    qual_vars: List[str]
    exposure_vars: List[str]
    finish_vars: List[str]
    garage_vars: List[str]
    drive_vars: List[str]
    fence_vars: List[str]
    categorical_vars: List[str]
    categorical_vars_label_encode: List[str]
    qual_mappings: Dict[str, int]
    fence_mappings: Dict[str, int]
    exposure_mappings: Dict[str, int]
    garage_mappings: Dict[str, int]
    finish_mappings: Dict[str, int]
    drive_mappings: Dict[str, int]


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_conf: ModelConfig


def create_and_validate_config() -> Config:
    """Run validation on config values."""

    #CONFIG_FILE_PATH = "regression_model/config.yml"

    with open(CONFIG_FILE_PATH, "r") as conf_file:
        parsed_config = load(conf_file.read())
    # print(parsed_config.data)

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_conf=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
# print(config)
