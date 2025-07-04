from regression_model.predict import predict_price

import numpy as np
import pandas as pd

def test_make_prediction(sample_input_data_ex):
    # Given
    #expected_first_prediction_value = 113422
    #expected_no_predictions = 1449

    # When
    result = predict_price(pd.DataFrame(sample_input_data_ex))

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    #assert result.get("errors") is None
    #assert len(predictions) == expected_no_predictions
    #assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=100)
