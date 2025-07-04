from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from regression_model.config.core import config
from regression_model.processing.features import (
    DropFeatures,
    FrequentMapper,
    LabelEncoderDef,
    LogMapper,
    MedianMapper,
    MissingMapper,
    OrdinalEncoder,
    PresentNa,
    RareLabelEncoder,
    TemporalMapper,
)

price_pipe = Pipeline(
    [
        ("medianImputer", MedianMapper(config.model_conf.numerical_vars_with_na)),
        ("frequencyImputer", FrequentMapper(config.model_conf.categorical_vars)),
        (
            "MissingImputer",
            MissingMapper(config.model_conf.categorical_vars_with_na_missing),
        ),
        (
            "TemporalMapper",
            TemporalMapper(config.model_conf.temporal_vars, config.model_conf.ref_var),
        ),
        ("DropFeatures", DropFeatures(config.model_conf.ref_var)),
        ("LogMapper", LogMapper(config.model_conf.numericals_log_vars)),
        (
            "qual_mapper",
            OrdinalEncoder(
                config.model_conf.qual_vars, config.model_conf.qual_mappings
            ),
        ),
        (
            "exposure_mapper",
            OrdinalEncoder(
                config.model_conf.exposure_vars, config.model_conf.exposure_mappings
            ),
        ),
        (
            "finish_mapper",
            OrdinalEncoder(
                config.model_conf.finish_vars, config.model_conf.finish_mappings
            ),
        ),
        (
            "garage_mapper",
            OrdinalEncoder(
                config.model_conf.garage_vars, config.model_conf.garage_mappings
            ),
        ),
        (
            "drive_mapper",
            OrdinalEncoder(
                config.model_conf.drive_vars, config.model_conf.drive_mappings
            ),
        ),
        (
            "fence_mapper",
            OrdinalEncoder(
                config.model_conf.fence_vars, config.model_conf.fence_mappings
            ),
        ),
        (
            "rareEncoder",
            RareLabelEncoder(config.model_conf.categorical_vars_label_encode),
        ),
        (
            "LabelMapper",
            LabelEncoderDef(config.model_conf.categorical_vars_label_encode),
        ),
        ("checkNa", PresentNa(config.model_conf.features)),
        ("scaler", MinMaxScaler()),
        (
            "lasso",
            Lasso(
                alpha=config.model_conf.alpha,
                random_state=config.model_conf.random_state,
            ),
        ),
    ]
)
