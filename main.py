import csv
import logging
import random
import time

from data.data_crafter import DataCrafter
from linear_regression import LinearRegressionWithGradientDescent
from metrics.metrics_crafter import MetricsCrafter


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
    spendings = DataCrafter.z_normalizer(spendings)  # !!!

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

    rr_metric = MetricsCrafter.r_squared(sales_test_fact, sales_test_forecast)
    mse_metric = MetricsCrafter.mean_squared_error(sales_test_fact, sales_test_forecast)
    mae_metric = MetricsCrafter.mean_absolute_error(sales_test_fact, sales_test_forecast)
    logging.info(f"Model testing results: R^2={rr_metric} MSE={mse_metric} MAE={mae_metric}")

    # Forecasting sales based on budget value
    adv_budget = 23.0  # spendings - 15 expected for 23
    # adv_budget = (adv_budget - ml_model.MEAN) / ml_model.STD_DEV  # normalization in case of data normalization
    sales_forecast = ml_model.predict(adv_budget, w, b)  # sales
    logging.info(
        f"Sales forecast will be: {round(sales_forecast, 1)} with advertising costs: {adv_budget}")  # 15.09 expected


if __name__ == "__main__":
    model_testing_launcher()
