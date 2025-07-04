from typing import Dict

import numpy as np
import pandas as pd

from regression_model import __version__ as _version
from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset, load_pipeline
from regression_model.processing.validation import validate_inputs

pipe_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
price_pipe = load_pipeline(pipe_name)


def predict_price(data: pd.DataFrame) -> Dict:

    validated_data = validate_inputs(data)
    print(validated_data)

    predict_ = price_pipe.predict(validated_data)

    result = {"predictions": [np.exp(a) for a in predict_], "version": _version}

    return result


if __name__ == "__main__":

    test_fname = f"{config.app_config.test_data_file}"
    test_df = load_dataset(test_fname)
    res_ = predict_price(test_df)
    print(res_)
