import numpy as np


class BinaryLogisticRegression:
    """
    Logistic Regression implementation
    """

    def __init__(self, descent_steps=1000, lr=0.001):
        self.weights = None
        self.descent_steps = descent_steps
        self._vsigmoid = np.vectorize(self._sigmoid)
        self.lr = lr

    def fit(self, x, y):
        if y.shape[0] != x.shape[0]:
            print("x and y shapes do not match")
            return

        x_ext = np.ones((x.shape[0], x.shape[1] + 1))
        x_ext[:, 1:] = x

        self.weights = 1 - np.random.rand(x.shape)

    def predict(self, x):
        if self.weights is None:
            print("You should fit data first with fit() function")
            return

        if x.shape[1] + 1 != self.weights.shape[0]:
            print("x shape does not match fitted data shape")
            return

        x_ext = np.ones((x.shape[0], x.shape[1] + 1))
        x_ext[:, 1:] = x

        pass

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def gradient_descent(self, x, y):
        for i in range(self.descent_steps):
            pass

    def loss(self, x, y):
        x_ext = np.ones((x.shape[0], x.shape[1] + 1))
        x_ext[:, 1:] = x

        x_one = np.log(self._vsigmoid(np.dot(self.weights, x_ext)))
        x_zero = np.log(1 - self._vsigmoid(np.dot(self.weights, x_ext)))

        y_one = np.multiply(y, x_one)
        y_zero = np.multiply(1 - y, x_zero)

        loss = -np.sum(y_one + y_zero)

        return loss

    def update_weights(self):
        pass
