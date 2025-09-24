import logging

# Setting up the logger for console output with INFO level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class LinearRegressionWithGradientDescent:

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
        # TODO switch to common notation X and Y
        dl_dw = 0.0  # Gradient accumulator for w
        dl_db = 0.0  # Gradient accumulator for b
        N = len(spendings)

        for i in range(N):
            prediction = w * spendings[i] + b
            error = sales[i] - prediction
            dl_dw += -2 * spendings[i] * error  # Gradient for w
            dl_db += -2 * error  # Gradient for b
            # TODO delete logging after finishing tests and upgrades
            # if i % 50 == 0:
            #     logging.info(f"i: {i} dw: {dl_dw} db: {dl_db}, adv: {spendings[i]} pred: {prediction} err: {error}")
            # logging.info("      i: {:3d} dw: {:12.3f} db: {:12.3f} adv: {:4.2f} pred: {:5.3f} err: {:8.8}".format(i, dl_dw, dl_db, spendings[i], prediction, error))
        # logging.info(f"dl_dw: {dl_dw} dl_db: {dl_db}")

        # Adding the regularization component
        # TODO check if regularization really needed
        dl_dw += 2 * tikhonov_regular * w  # for L2 regularization
        dl_db += 2 * tikhonov_regular * b  # for L2 regularization

        # Linear regularization is added to dl_db if necessary
        # TODO check if regularization really needed
        w -= (learning_rate / float(N)) * dl_dw
        b -= (learning_rate / float(N)) * dl_db
        logging.info(f"   Total ----------> w: {w} b: {b}")

        # Limiting the values of w and b to prevent overflow and underflow
        # TODO check if min max really needed
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
