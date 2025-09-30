class MetricsCrafter:

    @staticmethod
    def r_squared(y_fact: list[float], y_forecast: list[float]) -> float:
        """
        Computes the coefficient of determination R^2.
        The R² value ranges from 0 to 1, where 1 indicates that the model
        explains 100% of the variation, and 0 indicates that the model explains nothing.

        :param y_fact: A list of true values.
        :param y_forecast: A list of predicted values.
        :return: R^2 value [0...1] but NO! Value is between [-inf...1]
        """
        if not y_fact or not y_forecast:
            raise ValueError('y_fact neither y_forecast nor both can not be empty')
        if len(y_fact) != len(y_forecast):
            raise ValueError(f'Unequal dimension y_fact:{len(y_fact)} and y_forecast:{len(y_forecast)}')
        ss_total = sum((y - sum(y_fact) / len(y_fact)) ** 2 for y in y_fact)
        if ss_total == 0:
            return 1
        ss_residual = sum((y_fact[i] - y_forecast[i]) ** 2 for i in range(len(y_fact)))
        return 1 - ss_residual / ss_total

    @staticmethod
    def mean_absolute_error(y_fact: list[float], y_forecast: list[float]) -> float:
        """
        Computes the mean absolute error (MAE).
        It shows how much the model's predictions differ on average
        from the actual values, ignoring the direction of the error.

        :param y_fact: A list of true values.
        :param y_forecast: A list of predicted values.
        :return: Mean absolute error (MAE).
        """
        if not y_fact or not y_forecast:
            raise ValueError('y_fact neither y_forecast nor both can not be empty')
        if len(y_fact) != len(y_forecast):
            raise ValueError(f'Unequal dimension y_fact:{len(y_fact)} and y_forecast:{len(y_forecast)}')
        n = len(y_fact)
        return sum(abs(y_fact[i] - y_forecast[i]) for i in range(n)) / n

    @staticmethod
    def mean_squared_error(y_fact: list[float], y_forecast: list[float]) -> float:
        """
        Computes the mean squared error (MSE).
        MSE emphasizes large errors; the bigger the error,
        the greater its impact on the final metric value.

        :param y_fact: A list of true values.
        :param y_forecast: A list of predicted values.
        :return: Mean squared error (MSE).
        """
        if not y_fact or not y_forecast:
            raise ValueError('y_fact neither y_forecast nor both can not be empty')
        if len(y_fact) != len(y_forecast):
            raise ValueError(f'Unequal dimension y_fact:{len(y_fact)} and y_forecast:{len(y_forecast)}')
        n = len(y_fact)
        return sum((y_fact[i] - y_forecast[i]) ** 2 for i in range(n)) / n


# Example usage
if __name__ == "__main__":
    ...
