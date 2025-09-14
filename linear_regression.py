import csv
import logging
import time
import random

# Настройка логгера для вывода в консоль с уровнем INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class LinearRegressionWithGradientDescent:

    @staticmethod
    def data_normalizer(data: list[float]) -> list[float]:
        """
        Нормализация данных позволяет сделать их сравнимыми
        и улучшить сходимость алгоритма градиентного спуска.

        Нормализация помогает избежать ситуаций, когда признаки (входные значения)
        имеют разные масштабы, что может негативно сказаться на процессе обучения
        модели. Например, если один признак имеет значения в диапазоне от 0 до 1,
        а другой в диапазоне от 0 до 1000, то модель может отдать предпочтение
        более крупному признаку, игнорируя другие, что приведет к неправильному обучению.

        Нормализация осуществляется по формуле:
        normalized_value = (value - mean) / std_dev,

        где mean - среднее значение списка, а std_dev - стандартное отклонение.
        Если стандартное отклонение равно нулю (что может произойти, если все
        значения в списке одинаковы), функция возвращает список, заполненный нулями
        для избежания деления на ноль.

        :param data: Список числовых значений, которые требуется нормализовать.
        :return: Новый список, содержащий нормализованные значения.
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
            tikhonov_regular: float  # Параметр регуляризации
    ) -> tuple[float, float]:
        """
        Выполняет один шаг градиентного спуска для обновления параметров линейной модели y = w * x + b.

        :param spendings: список значений признака
        :param sales: список значений целевой переменной
        :param w: текущий коэффициент наклона модели
        :param b: текущий свободный член модели
        :param learning_rate: скорость обучения, шаг с которым мы двигаемся
        :param tikhonov_regular: параметр регуляризации, регуляризация Тихонова или L2 (Ridge) регуляризация
        :return: обновленные параметры (w, b)
        """
        dl_dw = 0.0  # накопитель градиента по w
        dl_db = 0.0  # накопитель градиента по b
        N = len(spendings)

        for i in range(N):
            prediction = w * spendings[i] + b
            error = sales[i] - prediction
            dl_dw += -2 * spendings[i] * error  # градиент для w
            dl_db += -2 * error  # градиент для b
            # if i % 50 == 0:
            #     logging.info(f"i: {i} dw: {dl_dw} db: {dl_db}, adv: {spendings[i]} pred: {prediction} err: {error}")
            # logging.info("      i: {:3d} dw: {:12.3f} db: {:12.3f} adv: {:4.2f} pred: {:5.3f} err: {:8.8}".format(i, dl_dw, dl_db, spendings[i], prediction, error))
        # logging.info(f"dl_dw: {dl_dw} dl_db: {dl_db}")

        # Добавляем регуляризационный компонент
        dl_dw += 2 * tikhonov_regular * w  # для L2 регуляризации
        dl_db += 2 * tikhonov_regular * b  # для L2 регуляризации

        # Линейная регуляризация добавляется к dl_db, если необходимо
        w -= (learning_rate / float(N)) * dl_dw
        b -= (learning_rate / float(N)) * dl_db
        logging.info(f"   Total ----------> w: {w} b: {b}")

        # Ограничение на значения w и b для предотвращения улета в стратосферу и переполнения
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
        Вычисляет среднеквадратичную ошибку (MSE) с учетом регуляризации.

        :param spendings: список значений признака
        :param sales: список значений целевой переменной
        :param w: коэффициент наклона модели
        :param b: свободный член модели
        :param regularization: параметр регуляризации
        :return: среднеквадратичная ошибка с учетом регуляризации
        """
        N = len(spendings)
        total_error = 0.0

        for i in range(N):
            prediction = w * spendings[i] + b
            error = sales[i] - prediction
            total_error += error ** 2

        # Добавление к ошибке регуляризационного штрафа
        total_error = (total_error / float(N)) + (regularization * w ** 2)  # L2 штраф

        return total_error

    def train(
            self,
            spendings: list[float],
            sales: list[float],
            w: float,
            b: float,
            learning_rate: float,
            epochs: int,
            regularization: float,  # Параметр регуляризации
            stop_factor: int = 0.00001,
    ) -> tuple[float, float]:
        """
        Обучает параметры линейной модели методом градиентного спуска с регуляризацией.

        :param spendings: список значений признака
        :param sales: список значений целевой переменной
        :param w: начальное значение коэффициента наклона
        :param b: начальное значение свободного члена
        :param learning_rate: скорость обучения
        :param epochs: количество итераций обучения
        :param regularization: параметр регуляризации
        :param stop_factor: уровень останова процесса оптимизации 0.00001 = 0.001%
        :return: обученные параметры (w, b)
        """
        previous_loss = None
        for epoch in range(epochs):
            w, b = self.update_w_and_b(spendings, sales, w, b, learning_rate, regularization)

            if epoch % 10 == 0:
                loss = self.avg_loss(spendings, sales, w, b, regularization)
                logging.info("Эпоха: {:5d} loss: {:4.10f} W={:8f} B={:8f}".format(epoch, loss, w, b))
                if previous_loss is not None:
                    if stop_factor > abs(loss - previous_loss) / max(previous_loss, 1e-10):  # prevent 0 division
                        break
                previous_loss = loss

        return w, b

    @staticmethod
    def predict(x: float, w: float, b: float) -> float:
        """
        Вычисляет предсказание линейной модели.

        :param x: входное значение
        :param w: коэффициент наклона
        :param b: свободный член
        :return: предсказанное значение
        """
        return w * x + b

    @staticmethod
    def r_squared(y_fact: list[float], y_forecast: list[float]) -> float:
        """
        Вычисляет коэффициент детерминации R^2.
        Значение R² варьируется от 0 до 1, где 1 указывает на то, что модель
        объясняет 100% вариации, а 0 указывает на то, что модель не объясняет ничего

        :param y_fact: Список истинных значений.
        :param y_forecast: Список предсказанных значений.
        :return: Значение R^2 [0...1]
        """
        ss_total = sum((y - sum(y_fact) / len(y_fact)) ** 2 for y in y_fact)
        ss_residual = sum((y_fact[i] - y_forecast[i]) ** 2 for i in range(len(y_fact)))
        r2 = 1 - (ss_residual / ss_total)
        return r2

    @staticmethod
    def mean_absolute_error(y_fact: list[float], y_forecast: list[float]) -> float:
        """
        Вычисляет среднюю абсолютную ошибку (MAE).
        Показывает, насколько в среднем предсказания модели отличаются
        от фактических значений, при этом игнорируя направление ошибки

        :param y_fact: Список истинных значений.
        :param y_forecast: Список предсказанных значений.
        :return: Средняя абсолютная ошибка (MAE).
        """
        n = len(y_fact)
        mae = sum(abs(y_fact[i] - y_forecast[i]) for i in range(n)) / n
        return mae

    @staticmethod
    def mean_squared_error(y_fact: list[float], y_forecast: list[float]) -> float:
        """
        Вычисляет среднюю квадратичную ошибку (MSE).
        MSE подчеркивает крупные ошибки; чем больше ошибка,
        тем больше ее влияние на итоговое значение метрики

        :param y_fact: Список истинных значений.
        :param y_forecast: Список предсказанных значений.
        :return: Средняя квадратичная ошибка (MSE).
        """
        n = len(y_fact)
        mse = sum((y_fact[i] - y_forecast[i]) ** 2 for i in range(n)) / n
        return mse


def create_data_set(path: str = 'advertising.csv', x: int = 0, y: int = 3) -> tuple[list, list]:
    """
    Чтение данных из CSV файла
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
            # отбираем 2 колонки = X -> реклама и Y -> продажи
            spendings.append(float(row[x]))
            sales.append(float(row[y]))
    return spendings, sales


def train_test_spliter(spendings: list, train_split_ratio=0.7) -> tuple[list, list]:
    """
    Разбиение датасета на тренировочную и тестовую выборки
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
    Песочника для запуска и тестирования модели
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

    # Тестовый запуск прогноза сущности y_new (sales - продажи) по значению x_new (spendings - затраты на рекламу)
    ml_model = LinearRegressionWithGradientDescent()

    # Попробуем нормализовать данные, чтобы избежать переобучения.
    # !!! Нормализация побеждает переобучение на данных advertising.csv колонки 0-3. Если её отключить будет беда!!!
    spendings = ml_model.data_normalizer(spendings)  # !!!

    # Создание среза данных для тренировки модели
    spendings_train_data = [spendings[i] for i in train_data_set_indexes]
    sales_train_data = [sales[i] for i in train_data_set_indexes]

    logging.info(
        f"Запуск тренировки LinReg модели на параметрах: w={w}, b={b} c alpha={learning_rate} на {iterations} итерациях и остановом на {stop_factor * 100}%")

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

    logging.info(f"Время выполнения тренировки модели: {round(end_time - start_time, 3)} секунд")
    logging.info(
        f"Правильнее - время вычисления значений критериев оптимизации или поиска значений коэффициентов 'W' и 'B' линейного уровнения y=wx+b")
    logging.info(f"Значения коэффициентов 'W' и 'B' линейного уровнения y=wx+b W = {w} B = {b}")

    # Тестирование результатов тренировки
    # Создание среза данных для тестирования полученной модели

    spendings_test = [spendings[i] for i in test_data_set_indexes]

    # Сортируем для большей наглядности и анализируемости результатов
    tuples = list(zip(spendings_test, test_data_set_indexes))
    tuples = sorted(tuples, key=lambda x: x)
    spendings_test = [i[0] for i in tuples]
    test_data_set_indexes = [i[1] for i in tuples]

    sales_test_fact = [sales[i] for i in test_data_set_indexes]
    sales_test_forecast = [ml_model.predict(val, w, b) for val in spendings_test]

    # Срез примеров данных Факт-Прогноз для визуального сравнения
    sample = 20
    assert sample <= len(sales_test_fact)
    for i in range(sample):
        logging.info(f'X={spendings_test[i]}, FACT={sales_test_fact[i]}, FORECAST={round(sales_test_forecast[i], 1)}')

    rr_metric = ml_model.r_squared(sales_test_fact, sales_test_forecast)
    mse_metric = ml_model.mean_squared_error(sales_test_fact, sales_test_forecast)
    mae_metric = ml_model.mean_absolute_error(sales_test_fact, sales_test_forecast)
    logging.info(f"Результаты тестирования модели: R^2={rr_metric} MSE={mse_metric} MAE={mae_metric}")

    # Прогнозирование продаж по значению бюджета
    adv_budget = 23.0  # spendings - 15.09 ожидается для 23
    # adv_budget = (adv_budget - ml_model.MEAN) / ml_model.STD_DEV  # нормализация в случае нормализации данных
    sales_forecast = ml_model.predict(adv_budget, w, b)  # sales
    logging.info(
        f"Прогноз продаж составит: {round(sales_forecast, 1)} при затратах на рекламу: {adv_budget}")  # 15.09 ожидается


if __name__ == "__main__":
    model_testing_launcher()
