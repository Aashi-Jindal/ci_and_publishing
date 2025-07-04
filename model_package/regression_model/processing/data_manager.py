from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from regression_model import __version__ as _version
from regression_model.config.core import config, DATASET_DIR, TRAINED_MODEL_DIR


def load_dataset(fname: str) -> pd.DataFrame:

    df = pd.read_csv(Path(f"{DATASET_DIR}/{fname}"))
    df.drop("Id", axis=1, inplace=True)
    df = df.rename(columns=config.model_conf.variables_to_rename)
    return df


def save_pipeline(pipe: Pipeline):

    pipe_path = Path(f"{TRAINED_MODEL_DIR}/{config.app_config.pipeline_save_file}{_version}.pkl")
    joblib.dump(pipe, pipe_path)


def load_pipeline(fname: str) -> Pipeline:

    pipe_path = Path(f"{TRAINED_MODEL_DIR}/{fname}")
    pipe_ = joblib.load(pipe_path)
    return pipe_
