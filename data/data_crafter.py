import logging
from statistics import mean, median


class DataCrafter:
    """
    Normalizing data actually looks like a big crutch
    Normalizing the data allows making it comparable and improving the convergence
    of the gradient descent algorithm.

    Normalization helps to avoid situations where the features (input values)
    have different scales, which can negatively affect the training process
    of the model. For example, if one feature has values in the range of 0 to 1,
    and another in the range of 0 to 1000, then the model may give preference to
    the larger feature, ignoring others, resulting in incorrect training.
    """

    @staticmethod
    def z_normalizer(data: list[float]) -> list[float]:
        """
        Standardisation is z-normalization
        Normalization is carried out using the formula: normalized_value = (value - mean) / std_dev,

        where mean is the average value of the list, and std_dev is the standard deviation.
        If the standard deviation equals zero (which can happen if all values in the list are the same),
        the function returns a list filled with zeros to avoid division by zero.

        :param data: A list of numerical values to be normalized.
        :return: A new list containing the normalized values.
        """
        if not data:
            logging.error(f"Got empty data to normalize in z_normalizer")
            return []
        mean = sum(data) / len(data)
        std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
        if std_dev == 0:  # zero division protection
            return [0.0 for _ in data]
        return [(x - mean) / std_dev for x in data]

    @staticmethod
    def min_max_normalizer(data: list[float]) -> list[float]:
        """
        Applies Min-Max normalization to the data into range from 0 to 1.
        :param data: A list of numerical values to normalize.
        :return: A list of normalized values.
        """
        if not data:
            logging.error(f"Got empty data to normalize in min_max_normalizer")
            return []
        min_value = min(data)  # Find the minimum value in the data
        max_value = max(data)  # Find the maximum value in the data
        delta = max_value - min_value  # Find base

        # Avoid division by zero when max_value equals min_value
        if max_value == min_value:
            return [0.0 for _ in data]
        # Apply Min-Max normalization
        return [(x - min_value) / delta for x in data]

    @staticmethod
    def aberration_normalizer(data: list[float]) -> list[float]:
        """
        Normalization based on median as 1 and range from 0 to inf
        :param data: A list of numerical values to normalize.
        :return: A list of normalized values.
        """
        if not data:
            logging.error(f"Got empty data to normalize in aberration_normalizer")
            return []
        median_value = median(data)
        min_value = min(data)
        delta = median_value - min_value
        # Avoid division by zero when median_value equals min_value
        if median_value == min_value:
            return [0.0 for _ in data]
        return [(x - min_value) / delta for x in data]

    @staticmethod
    def waterline_normalizer(data: list[float]) -> list[float]:
        """
        Pushes up values from subzero to above zero and via versa based on weight of values in data.
        Looks a-like z normalization
        :param data: A list of numerical values to normalize.
        :return: A list of normalized values.
        """
        if not data:
            logging.error(f"Got empty data to normalize in waterline_normalizer")
            return []
        mean_value = mean(data)  # Calculate the mean value
        normalized_data = [(x - mean_value) for x in data]  # Center the data around the mean
        max_abs_value = max(abs(min(data)), abs(max(data)))  # Ensure we scale with the absolute maximum
        if max_abs_value == 0:
            return [0.0 for _ in data]  # Handle case if all values are zero

        return [x / max_abs_value for x in normalized_data]

    @staticmethod
    def minus_one_to_one_normalizer(data: list[float]) -> list[float]:
        """
        Applies Min-Max normalization to the data, scaling it to the range [-1, 1].

        :param data: A list of numerical values to normalize.
        :return: A list of normalized values scaled to the range [-1, 1].
        """
        if not data:
            logging.error(f"Got empty data to normalize in minus_one_to_one_normalizer")
            return []
        min_value = min(data)  # Find the minimum value in the data
        max_value = max(data)  # Find the maximum value in the data
        delta = max_value - min_value

        # Avoid division by zero when max_value equals min_value
        if max_value == min_value:
            return [0.0 for _ in data]

        # Apply Min-Max normalization to the range [-1, 1]
        normalized_data = [((x - min_value) / delta) * 2 - 1 for x in data]
        return normalized_data


# Example usage
if __name__ == "__main__":
    # Setting up the logger for console output with INFO level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Original data chunks
    fixed = [1,1,1,1,1,1,1]
    mix_data = [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 300]
    symmetric_zero = [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50]
    start_zero = [0, 10, 20, 30, 40, 50, 300]
    finish_zero = [-300, -50, -40, -30, -20, -10, 0]
    above_zero = [10, 20, 30, 40, 50, 300]
    wide_range = [-10**6, -11, -5, 0, 5, 100, 1000, 10**20]

    full_data_set = (fixed, mix_data, symmetric_zero, start_zero, finish_zero, above_zero, wide_range)

    def out_put_normalizer(data: list):
        """
        Do not use big data sets because visualization will be lost
        :param data: data[int|float]
        :return: None
        """
        logging.info(f"Original data: {data}")
        # add new methods as pairs
        normalization_methods = [
            ('Z Normalization', DataCrafter.z_normalizer),
            ('Min-Max Normalization', DataCrafter.min_max_normalizer),
            ('Aberration Normalization', DataCrafter.aberration_normalizer),
            ('Waterline Normalization', DataCrafter.waterline_normalizer),
            ('Minus-One to One Normalization', DataCrafter.minus_one_to_one_normalizer)
        ]
        for index, pair in enumerate(normalization_methods):
            method_name, method = pair
            normalized_data = method(data)
            logging.info(f"{method_name} normalized data looks like: {[round(i, 3) for i in normalized_data]}")


    for index, item in enumerate(full_data_set):
        logging.info("\n")
        logging.info(f"Test #{index}")
        out_put_normalizer(item)
