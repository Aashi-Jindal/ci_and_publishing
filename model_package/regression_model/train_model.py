import numpy as np
from sklearn.model_selection import train_test_split

from regression_model.config.core import config
from regression_model.pipeline import price_pipe
from regression_model.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:

    data = load_dataset(config.app_config.training_data_file)

    X_train, X_test, Y_train, Y_test = train_test_split(
        data[config.model_conf.features],
        data[config.model_conf.target],
        test_size=config.model_conf.test_size,
        random_state=config.model_conf.random_state,
    )

    y_train = np.log(Y_train)

    price_pipe.fit(X_train, y_train)

    save_pipeline(price_pipe)


if __name__ == "__main__":

    run_training()
