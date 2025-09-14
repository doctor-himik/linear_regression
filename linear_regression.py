import csv
import logging
import time
import random

# Setting up the logger for console output with INFO level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class LinearRegressionWithGradientDescent:

    @staticmethod
    def data_normalizer(data: list[float]) -> list[float]:
        """
        Normalizing data actually looks like a crutch
        Normalizing the data allows making it comparable and improving the convergence of the gradient descent algorithm.

        Normalization helps to avoid situations where the features (input values)
        have different scales, which can negatively affect the training process
        of the model. For example, if one feature has values in the range of 0 to 1,
        and another in the range of 0 to 1000, then the model may give preference to
        the larger feature, ignoring others, resulting in incorrect training.

        Normalization is carried out using the formula: normalized_value = (value - mean) / std_dev,

        where mean is the average value of the list, and std_dev is the standard deviation.
        If the standard deviation equals zero (which can happen if all
        values in the list are the same), the function returns a list filled with zeros
        to avoid division by zero.

        :param data: A list of numerical values to be normalized.
        :return: A new list containing the normalized values.
        """
        mean = sum(data) / len(data)
        std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
        if std_dev == 0:  # zero division protection
            return [0 for _ in data]
        return [(x - mean) / std_dev for x in data]

    @staticmethod
    def update_w_and_b(
            spendings: list[float],
            sales: list[float],
            w: float,
            b: float,
            learning_rate: float,
            tikhonov_regular: float  # Regularization parameter
    ) -> tuple[float, float]:
        """
        Performs one step of gradient descent to update the parameters of the linear model y = w * x + b.

        :param spendings: A list of feature values
        :param sales: A list of target variable values
        :param w: The current slope coefficient of the model
        :param b: The current intercept of the model
        :param learning_rate: The learning rate, the step we take
        :param tikhonov_regular: Regularization parameter, regularization Тихонова or L2 (Ridge) regularization
        :return: Updated parameters (w, b)
        """
        # TODO switch to common notation X Y
        dl_dw = 0.0  # Gradient accumulator for w
        dl_db = 0.0  # Gradient accumulator for b
        N = len(spendings)

        for i in range(N):
            prediction = w * spendings[i] + b
            error = sales[i] - prediction
            dl_dw += -2 * spendings[i] * error  # Gradient for w
            dl_db += -2 * error  # Gradient for b
            # TODO fix trash
            # if i % 50 == 0:
            #     logging.info(f"i: {i} dw: {dl_dw} db: {dl_db}, adv: {spendings[i]} pred: {prediction} err: {error}")
            # logging.info("      i: {:3d} dw: {:12.3f} db: {:12.3f} adv: {:4.2f} pred: {:5.3f} err: {:8.8}".format(i, dl_dw, dl_db, spendings[i], prediction, error))
        # logging.info(f"dl_dw: {dl_dw} dl_db: {dl_db}")

        # Adding the regularization component
        dl_dw += 2 * tikhonov_regular * w  # for L2 regularization
        dl_db += 2 * tikhonov_regular * b  # for L2 regularization

        # Linear regularization is added to dl_db if necessary
        w -= (learning_rate / float(N)) * dl_dw
        b -= (learning_rate / float(N)) * dl_db
        logging.info(f"   Total ----------> w: {w} b: {b}")

        # Limiting the values of w and b to prevent overflow and underflow
        w = max(min(w, 1e10), -1e10)
        b = max(min(b, 1e10), -1e10)

        logging.info(f"      Fixed ----------> w: {w} b: {b}")
        return w, b

    @staticmethod
    def avg_loss(
            spendings: list[float],
            sales: list[float],
            w: float,
            b: float,
            regularization: float,
    ) -> float:
        """
        Computes the mean squared error (MSE) considering regularization.

        :param spendings: A list of feature values
        :param sales: A list of target variable values
        :param w: Slope coefficient of the model
        :param b: Intercept of the model
        :param regularization: Regularization parameter
        :return: Mean squared error considering regularization
        """
        N = len(spendings)
        total_error = 0.0

        for i in range(N):
            prediction = w * spendings[i] + b
            error = sales[i] - prediction
            total_error += error ** 2

        # Adding the regularization penalty to the error
        total_error = (total_error / float(N)) + (regularization * w ** 2)  # L2 penalty

        return total_error

    def train(
            self,
            spendings: list[float],
            sales: list[float],
            w: float,
            b: float,
            learning_rate: float,
            epochs: int,
            regularization: float,  # Regularization parameter
            stop_factor: int = 0.00001,
    ) -> tuple[float, float]:
        """
        Trains the parameters of the linear model using gradient descent with regularization.

        :param spendings: A list of feature values
        :param sales: A list of target variable values
        :param w: Initial slope coefficient
        :param b: Initial intercept
        :param learning_rate: Learning rate
        :param epochs: Number of training iterations
        :param regularization: Regularization parameter
        :param stop_factor: Stopping factor for optimization process 0.00001 = 0.001%
        :return: Trained parameters (w, b)
        """
        previous_loss = None
        for epoch in range(epochs):
            w, b = self.update_w_and_b(spendings, sales, w, b, learning_rate, regularization)

            if epoch % 10 == 0:
                loss = self.avg_loss(spendings, sales, w, b, regularization)
                logging.info("Epoch: {:5d} loss: {:4.10f} W={:8f} B={:8f}".format(epoch, loss, w, b))
                if previous_loss is not None:
                    if stop_factor > abs(loss - previous_loss) / max(previous_loss, 1e-10):  # prevent 0 division
                        break
                previous_loss = loss

        return w, b

    @staticmethod
    def predict(x: float, w: float, b: float) -> float:
        """
        Computes the prediction of the linear model.

        :param x: Input value
        :param w: Slope coefficient
        :param b: Intercept
        :return: Predicted value
        """
        return w * x + b

    @staticmethod
    def r_squared(y_fact: list[float], y_forecast: list[float]) -> float:
        """
        Computes the coefficient of determination R^2.
        The R² value ranges from 0 to 1, where 1 indicates that the model
        explains 100% of the variation, and 0 indicates that the model explains nothing.

        :param y_fact: A list of true values.
        :param y_forecast: A list of predicted values.
        :return: R^2 value [0...1]
        """
        ss_total = sum((y - sum(y_fact) / len(y_fact)) ** 2 for y in y_fact)
        ss_residual = sum((y_fact[i] - y_forecast[i]) ** 2 for i in range(len(y_fact)))
        r2 = 1 - (ss_residual / ss_total)
        return r2

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
        n = len(y_fact)
        mae = sum(abs(y_fact[i] - y_forecast[i]) for i in range(n)) / n
        return mae

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
        n = len(y_fact)
        mse = sum((y_fact[i] - y_forecast[i]) ** 2 for i in range(n)) / n
        return mse


def create_data_set(path: str = 'data/advertising.csv', x: int = 0, y: int = 3) -> tuple[list, list]:
    """
    Reading data from a CSV file
    :param y: target
    :param x: data
    :param path: str path to the file with data
    :return: tuple[list, list]
    """
    spendings = []
    sales = []
    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0].isalpha():
                continue
            # Selecting 2 columns = X -> advertisement and Y -> sales
            spendings.append(float(row[x]))
            sales.append(float(row[y]))
    return spendings, sales


def train_test_spliter(spendings: list, train_split_ratio=0.7) -> tuple[list, list]:
    """
    Splitting the dataset into training and test subsets
    :param spendings: list full dataset
    :param train_split_ratio: % amount of data for train purposes
    :return: tuple[list, list]
    """
    train_data_set_indexes = []
    test_data_set_indexes = []

    # create list of indexes for 'test_data_set_indexes'
    for i in range(int(len(spendings) * (1 - train_split_ratio))):
        while True:
            index = random.randint(0, len(spendings) - 1)
            if index not in test_data_set_indexes:
                test_data_set_indexes.append(index)
                break

    # filter list of indexes for 'train_data_set_indexes'
    for i in range(int(len(spendings))):
        if i not in test_data_set_indexes:
            train_data_set_indexes.append(i)

    assert len(train_data_set_indexes) + len(test_data_set_indexes) == len(spendings), 'Check data set split logics'
    return train_data_set_indexes, test_data_set_indexes


def model_testing_launcher(
        learning_rate=0.001,
        iterations=15000,
        w=0.0,
        b=0.0,
        stop_factor=0.00001,
        ridge=0.001,
):
    """
    Sandbox for launching and testing the model
    :param learning_rate:
    :param iterations:
    :param w:
    :param b:
    :param stop_factor:
    :param ridge:
    :return:
    """
    # Read data
    spendings, sales = create_data_set(x=0, y=3)
    # Split data
    train_data_set_indexes, test_data_set_indexes = train_test_spliter(spendings, train_split_ratio=0.8)

    # Test launch forecast entity y_new (sales) by value x_new (spendings)
    ml_model = LinearRegressionWithGradientDescent()

    # Let's try normalizing data to avoid overfitting.
    # !!! Normalization prevents overfitting on the data advertising.csv columns 0-3. If you disable it, there will be problems!!!
    spendings = ml_model.data_normalizer(spendings)  # !!!

    # Creating a data slice for training the model
    spendings_train_data = [spendings[i] for i in train_data_set_indexes]
    sales_train_data = [sales[i] for i in train_data_set_indexes]

    logging.info(
        f"Launching LinReg model training with parameters: w={w}, b={b} learning rate={learning_rate} for {iterations} iterations and stopping at {stop_factor * 100}%")

    start_time = time.time()
    w, b = ml_model.train(
        spendings=spendings_train_data,
        sales=sales_train_data,
        w=w,
        b=b,
        learning_rate=learning_rate,
        epochs=iterations,
        regularization=ridge,
    )
    end_time = time.time()

    logging.info(f"Training model took: {round(end_time - start_time, 3)} seconds")
    logging.info(
        f"Right - time to compute optimization criteria or to find values of coefficients 'W' and 'B' of the linear equation y=wx+b")
    logging.info(f"Values of coefficients 'W' and 'B' of the linear equation y=wx+b W = {w} B = {b}")

    # Testing the training results
    # Creating a data slice for testing the obtained model

    spendings_test = [spendings[i] for i in test_data_set_indexes]

    # Sorting for better visibility and analyzability of results
    tuples = list(zip(spendings_test, test_data_set_indexes))
    tuples = sorted(tuples, key=lambda x: x)
    spendings_test = [i[0] for i in tuples]
    test_data_set_indexes = [i[1] for i in tuples]

    sales_test_fact = [sales[i] for i in test_data_set_indexes]
    sales_test_forecast = [ml_model.predict(val, w, b) for val in spendings_test]

    # Sample of data examples Факт-Прогноз for visual comparison
    sample = 20
    assert sample <= len(sales_test_fact)
    for i in range(sample):
        logging.info(f'X={spendings_test[i]}, FACT={sales_test_fact[i]}, FORECAST={round(sales_test_forecast[i], 1)}')

    rr_metric = ml_model.r_squared(sales_test_fact, sales_test_forecast)
    mse_metric = ml_model.mean_squared_error(sales_test_fact, sales_test_forecast)
    mae_metric = ml_model.mean_absolute_error(sales_test_fact, sales_test_forecast)
    logging.info(f"Model testing results: R^2={rr_metric} MSE={mse_metric} MAE={mae_metric}")

    # Forecasting sales based on budget value
    adv_budget = 23.0  # spendings - 15 expected for 23
    # adv_budget = (adv_budget - ml_model.MEAN) / ml_model.STD_DEV  # normalization in case of data normalization
    sales_forecast = ml_model.predict(adv_budget, w, b)  # sales
    logging.info(
        f"Sales forecast will be: {round(sales_forecast, 1)} with advertising costs: {adv_budget}")  # 15.09 expected


if __name__ == "__main__":
    model_testing_launcher()
