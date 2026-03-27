import numpy as np
import utils


class GaussianNaiveBayes:
    def __init__(self):
        self.means = None
        self.vars = None
        self.priors = None

    def fit(self, X, y):
        # Convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Extract classes
        all_classes = np.unique(y)
        print(all_classes)

        # Compute class priors
        self.priors = {}
        for cls in all_classes:
            self.priors[cls] = np.sum(y == cls) / len(y)
        print(self.priors)

        # Compute mean and variance per feature per class
        self.means = {}
        self.vars = {}
        for cls in all_classes:
            X_cls = X[y == cls]  # Filter data for the current class
            self.means[cls] = [utils.compute_mean(X_cls[:, feature]) for feature in range(X_cls.shape[1])]
            self.vars[cls] = [utils.compute_var(X_cls[:, feature]) for feature in range(X_cls.shape[1])]
        print(self.means)
        print(self.vars)

    def log_likelihood(self, X, y):
        pass

    def predict(self, X):
        pass

# Dummy data
X = [[1, 2], [3, 4], [5, 6]]
y = ['A', 'B', 'A']
model = GaussianNaiveBayes()
model.fit(X, y)
