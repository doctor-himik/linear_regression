import pytest

from data.data_crafter import DataCrafter


class TestDataCrafter:
    # test data
    fixed_one = [1, 1, 1, 1, 1, 1, 1]
    mix_data = [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 300]
    symmetric_zero = [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50]
    start_zero = [0, 10, 20, 30, 40, 50, 300]
    finish_zero = [-300, -50, -40, -30, -20, -10, 0]
    above_zero = [10, 20, 30, 40, 50, 300]
    # TODO Rounded by 3 affects testing accuracy on big numbers like 10**6 etc.
    wide_range = [-10 ** 6, -11, -5, 0, 5, 100, 1000, 10 ** 20]
    negative = []
    corner_case = [5]

    @pytest.mark.unit
    @pytest.mark.parametrize("data, expected", [
        (corner_case, corner_case),
        (negative, []),
        (finish_zero, [-2.417, 0.146, 0.249, 0.352, 0.454, 0.557, 0.659]),
        (fixed_one, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (start_zero, [-0.659, -0.557, -0.454, -0.352, -0.249, -0.146, 2.417]),
        (above_zero, [-0.641, -0.542, -0.444, -0.345, -0.246, 2.218]),
        (symmetric_zero, [-1.508, -1.206, -0.905, -0.603, -0.302, 0.302, 0.603, 0.905, 1.206, 1.508]),
        (mix_data, [-0.841, -0.732, -0.623, -0.515, -0.406, -0.188, -0.079, 0.03, 0.139, 0.247, 2.969]),
        (wide_range, [-0.378, -0.378, -0.378, -0.378, -0.378, -0.378, -0.378, 2.646]),
    ])
    def test_z_normalizer(self, data, expected):
        normalized = DataCrafter.z_normalizer(data)
        for index in range(len(data)):
            assert normalized[index] - expected[index] < 0.001, 'Rounded by 3 after point you can change it'

    @pytest.mark.unit
    @pytest.mark.parametrize("data, expected", [
        (corner_case, corner_case),
        (negative, []),
        (finish_zero, [-0.786, 0.048, 0.081, 0.114, 0.148, 0.181, 0.214]),
        (fixed_one, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (above_zero, [-0.217, -0.183, -0.15, -0.117, -0.083, 0.75]),
        (start_zero, [-0.214, -0.181, -0.148, -0.114, -0.081, -0.048, 0.786]),
        (symmetric_zero, [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]),
        (mix_data, [-0.258, -0.224, -0.191, -0.158, -0.124, -0.058, -0.024, 0.009, 0.042, 0.076, 0.909]),
        (wide_range, [-0.125, -0.125, -0.125, -0.125, -0.125, -0.125, -0.125, 0.875])
    ])
    def test_waterline_normalizer(self, data, expected):
        """
        waterline_normalizer looks like a z_normalizer
        z_normalizer make the same range as waterline_normalizer but ~3 times wider
        z_normalizer[-1] - z_normalizer[0] ~= 3*(waterline_normalizer[-1] - waterline_normalizer[0])
        strange but looks interesting
        :param data: test data set
        :param expected: expected result
        :return: bool
        """
        normalized = DataCrafter.waterline_normalizer(data)
        for index in range(len(data)):
            assert normalized[index] - expected[index] < 0.001, 'Rounded by 3 after point you can change it'

    @pytest.mark.unit
    @pytest.mark.parametrize("data, expected", [
        (corner_case, corner_case),
        (negative, []),
        (finish_zero, [0.0, 0.833, 0.867, 0.9, 0.933, 0.967, 1.0]),
        (fixed_one, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (start_zero, [0.0, 0.033, 0.067, 0.1, 0.133, 0.167, 1.0]),
        (above_zero, [0.0, 0.034, 0.069, 0.103, 0.138, 1.0]),
        (symmetric_zero, [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]),
        (mix_data, [0.0, 0.029, 0.057, 0.086, 0.114, 0.171, 0.2, 0.229, 0.257, 0.286, 1.0]),
        (wide_range, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    ])
    def test_min_max_normalizer(self, data, expected):
        normalized = DataCrafter.min_max_normalizer(data)
        for index in range(len(data)):
            assert 0 <= normalized[index] <= 1, 'Value have to be in range [0,1]'
            assert normalized[index] - expected[index] < 0.001, 'Rounded by 3 after point you can change it'

    @pytest.mark.unit
    @pytest.mark.parametrize("data, expected", [
        (corner_case, corner_case),
        (negative, []),
        (finish_zero, [0.0, 0.926, 0.963, 1.0, 1.037, 1.074, 1.111]),
        (fixed_one, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (start_zero, [0.0, 0.333, 0.667, 1.0, 1.333, 1.667, 10.0]),
        (above_zero, [0.0, 0.4, 0.8, 1.2, 1.6, 11.6]),
        (symmetric_zero, [0.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8, 2.0]),
        (mix_data, [0.0, 0.167, 0.333, 0.5, 0.667, 1.0, 1.167, 1.333, 1.5, 1.667, 5.833]),
        (wide_range, [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.001, 99999750000626.0])
    ])
    def test_aberration_normalizer(self, data, expected):
        normalized = DataCrafter.aberration_normalizer(data)
        for index in range(len(data)):
            assert normalized[index] >= 0, 'all values have to be above zero'
            assert normalized[index] - expected[index] < 0.001, 'Rounded by 3 after point you can change it'

    @pytest.mark.unit
    @pytest.mark.parametrize("data, expected", [
        (corner_case, corner_case),
        (negative, []),
        (finish_zero, [-1.0, 0.667, 0.733, 0.8, 0.867, 0.933, 1.0]),
        (fixed_one, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (above_zero, [-1.0, -0.931, -0.862, -0.793, -0.724, 1.0]),
        (start_zero, [-1.0, -0.933, -0.867, -0.8, -0.733, -0.667, 1.0]),
        (symmetric_zero, [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]),
        (mix_data, [-1.0, -0.943, -0.886, -0.829, -0.771, -0.657, -0.6, -0.543, -0.486, -0.429, 1.0]),
        (wide_range, [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0])
    ])
    def test_minus_one_to_one_normalizer(self, data, expected):
        normalized = DataCrafter.minus_one_to_one_normalizer(data)
        for index in range(len(data)):
            assert normalized[index] - expected[index] < 0.001, 'Rounded by 3 after point you can change it'
