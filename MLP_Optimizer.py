import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV

class MLP_Optimizer:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.clf = None

    def optimize_mlp(self, x_train, y_train, x_test, y_test):
        param_space = {
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
            'tol': np.logspace(-4, -1, 4),
            'momentum': np.linspace(0.1, 0.9, 9)
        }

        mlp = MLPRegressor(max_iter=100)

        self.clf = RandomizedSearchCV(mlp, param_space, n_jobs=-1, cv=3)

        self.clf.fit(x_train, y_train)

        if self.verbose:
            print("Best parameters found: ", self.clf.best_params_)

        y_pred = self.clf.predict(x_test)

        train_score = self.clf.score(x_train, y_train)
        if self.verbose:
            print("Training score: ", train_score)

        test_score = self.clf.score(x_test, y_test)
        if self.verbose:
            print("Test score: ", test_score)

        plt.figure(figsize=(12, 6))
        plt.plot(y_test, color='blue', label='Actual')
        plt.plot(y_pred, color='red', label='Predicted')
        plt.title('Actual vs Predicted')
        plt.legend()
        plt.show()

        return y_pred