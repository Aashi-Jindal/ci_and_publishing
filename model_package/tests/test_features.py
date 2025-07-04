from regression_model.config.core import config
from regression_model.processing.features import TemporalMapper

import pandas as pd


def test_temporal_var_transformer(sample_input_data):
    # when
    tm = TemporalMapper(config.model_conf.temporal_vars, config.model_conf.ref_var)
    res = tm.fit_transform(sample_input_data)
    # then
    yrbuilt_obt = res["YearBuilt"].iat[0]
    assert yrbuilt_obt == 49  # (2010-1961)
