import pandas as pd

from regression_model.config.core import config


def validate_inputs(data: pd.DataFrame) -> pd.DataFrame:

    validated_data = data.copy()
    validated_data.rename(columns=config.model_conf.variables_to_rename, inplace=True)
    validated_data = validated_data[config.model_conf.features]

    new_na_vars = [
        var
        for var in config.model_conf.features
        if var not in config.model_conf.categorical_vars
        and var not in config.model_conf.categorical_vars_with_na_missing
        and var not in config.model_conf.numerical_vars_with_na
        and data[var].isna().sum() > 0
    ]

    validated_data.dropna(subset=new_na_vars, inplace=True)

    return validated_data
