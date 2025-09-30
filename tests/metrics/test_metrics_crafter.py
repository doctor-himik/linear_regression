from random import randint

import pytest

from metrics.metrics_crafter import MetricsCrafter

# Test data set
data_sets = [
    ([1, 2, 3], [1, 2, 3], (1.0, 0.0, 0.0)),  # Perfect prediction
    ([1, 2, 3], [1, 2, 2], (0.5, 0.3333, 0.3333)),  # Close prediction
    ([1, 1, 1], [1, 1, 1], (1.0, 0.0, 0.0)),  # All same values
    ([10, 20, 30], [15, 25, 35], (0.625, 5.0, 25.0)),  # Some error
    ([1, 2, 3], [4, 5, 6], (-12.5, 3.0, 9.0)),  # Poor prediction
]


class TestMetricsCrafter:
    @pytest.mark.parametrize("y_fact, y_forecast, expected", data_sets)
    def test_metrics_r_squared(self, y_fact, y_forecast, expected):
        # Проверка R^2
        r2_result = MetricsCrafter.r_squared(y_fact, y_forecast)
        assert round(r2_result, 4) == round(expected[0], 4)

    @pytest.mark.parametrize("y_fact, y_forecast, expected", data_sets)
    def test_metrics_mean_absolute_error(self, y_fact, y_forecast, expected):
        # Проверка MAE
        mae_result = MetricsCrafter.mean_absolute_error(y_fact, y_forecast)
        assert round(mae_result, 4) == round(expected[1], 4)

    @pytest.mark.parametrize("y_fact, y_forecast, expected", data_sets)
    def test_metrics_mean_squared_error(self, y_fact, y_forecast, expected):
        # Проверка MSE
        mse_result = MetricsCrafter.mean_squared_error(y_fact, y_forecast)
        assert round(mse_result, 4) == round(expected[2], 4)

    empty_data_set = [
        ([], []),
        ([1], []),
        ([], [1]),
        ([1, 2], [1]),
        ([1], [1, 2]),
    ]

    @pytest.mark.parametrize("y_fact, y_forecast", empty_data_set)
    def test_empty_lists_mean_absolute_error(self, y_fact, y_forecast):
        # Проверка обработки пустых списков
        with pytest.raises(ValueError):
            MetricsCrafter.mean_absolute_error(y_fact, y_forecast)

    @pytest.mark.parametrize("y_fact, y_forecast", empty_data_set)
    def test_empty_lists_mean_squared_error(self, y_fact, y_forecast):
        # Проверка обработки пустых списков
        with pytest.raises(ValueError):
            MetricsCrafter.mean_squared_error(y_fact, y_forecast)

    @pytest.mark.parametrize("y_fact, y_forecast", empty_data_set)
    def test_empty_lists_r_squared(self, y_fact, y_forecast):
        # Проверка обработки пустых списков
        with pytest.raises(ValueError):
            MetricsCrafter.r_squared(y_fact, y_forecast)

    def test_single_value_lists_r_squared(self):
        # Проверка обработки списков, содержащих одно значение
        y_fact = [5]
        y_forecast = [5]
        assert MetricsCrafter.r_squared(y_fact, y_forecast) == 1.0

    def test_single_value_lists_mean_absolute_error(self):
        # Проверка обработки списков, содержащих одно значение
        y_fact = [5]
        y_forecast = [5]
        assert MetricsCrafter.mean_absolute_error(y_fact, y_forecast) == 0.0

    def test_single_value_lists_mean_squared_error(self):
        # Проверка обработки списков, содержащих одно значение
        y_fact = [5]
        y_forecast = [5]
        assert MetricsCrafter.mean_squared_error(y_fact, y_forecast) == 0.0

    pbt_data_set = [
        ([1, 1, 1], [0, 0, 0]),  # Completely wrong predictions
        ([1, 1, 1], [1, 1, 1]),  # Completely right predictions
        ([1, 1, 1], [1, 1, 0]),  # More or less good predictions
        ([3, -3, 2.5], [-3, 3, -2.5]),  # Swapped predictions
        *tuple(
            (
                [randint(-1000, 1000) for i in range(4)], [randint(-1000, 1000) for q in range(4)])
            for x in range(100)
        ),  # 100 random fact - prediction
    ]

    @pytest.mark.parametrize("y_fact, y_forecast", pbt_data_set)
    def test_pbt_mean_absolute_error(self, y_fact, y_forecast):
        # Property-based tests for metrics
        assert MetricsCrafter.mean_absolute_error(y_fact, y_forecast) >= 0

    @pytest.mark.parametrize("y_fact, y_forecast", pbt_data_set)
    def test_pbt_mean_squared_error(self, y_fact, y_forecast):
        # Property-based tests for metrics
        assert MetricsCrafter.mean_squared_error(y_fact, y_forecast) >= 0

    @pytest.mark.parametrize("y_fact, y_forecast", pbt_data_set)
    def test_pbt_r_squared(self, y_fact, y_forecast):
        # Property-based tests for metrics - r_squared have to be in range [0,1]
        minus_inf = -10 ** 999  # fake minus inf just good enough for simple testing
        assert minus_inf <= MetricsCrafter.r_squared(y_fact, y_forecast) <= 1


if __name__ == "__main__":
    pytest.main()
