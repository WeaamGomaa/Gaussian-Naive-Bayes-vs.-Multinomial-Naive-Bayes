import numpy as np
import utils_funs


class GaussianNaiveBayes:
    def __init__(self):
        self.means = None
        self.vars = None
        self.priors = None
        self.classes = None
        self.class_probs = None

    def fit(self, X, y):
        # Convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Extract classes
        self.classes = np.unique(y)

        # Compute class priors
        self.priors = {}
        for cls in self.classes:
            self.priors[cls] = np.sum(y == cls) / len(y)

        # Compute mean and variance per feature per class
        self.means = {}
        self.vars = {}
        for cls in self.classes:
            X_cls = X[y == cls]  # Filter data for the current class
            self.means[cls] = [utils_funs.compute_mean(X_cls[:, feature]) for feature in range(X_cls.shape[1])]
            self.vars[cls] = [utils_funs.compute_var(X_cls[:, feature]) for feature in range(X_cls.shape[1])]

    def calculate_likelihood(self, x, mean, var):
        # Calculate the probability of a data point under Gaussian distribution
        eps = 1e-9  # Add a small value to the variance to prevent division by zero
        coefficient = 1 / np.sqrt(2 * np.pi * (var + eps))
        exponent = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
        return coefficient * exponent

    def calculate_log_likelihood(self, x, mean, var):
        eps = 1e-9
        return -0.5 * np.log(2 * np.pi * (var + eps)) - (0.5 * ((x - mean) ** 2) / (var + eps))

    def predict(self, X, use_log=True):
        # Convert the test sample to a numpy array
        X = np.array(X)
        predictions = []

        self.class_probs = []
        for sample in X:
            probs = {}
            for cls in self.classes:
                if use_log:
                    total_prob =  np.log(self.priors[cls])
                else:
                    total_prob = self.priors[cls]

                # Multiply by the likelihood of each feature
                for i in range(len(sample)):
                    feature_value = sample[i]
                    mean = self.means[cls][i]
                    var = self.vars[cls][i]

                    if use_log:
                        total_prob += self.calculate_log_likelihood(feature_value, mean, var)
                    else:
                        total_prob *= self.calculate_likelihood(feature_value, mean, var)

                probs[cls] = total_prob

            # Store the probabilities of each class for each sample
            self.class_probs.append(probs)

            # Get the class with the highest probability
            best_class = max(probs, key=probs.get)
            predictions.append(best_class)
        return predictions

    def predict_prob(self, X, use_log=True):
        X = np.array(X)
        self.predict(X, use_log)
        result = []
        for probs in self.class_probs:
            # Convert log probs back to real probs before normalizing
            if use_log:
                exp_probs = {cls: np.exp(prob) for cls, prob in probs.items()}
                probs = exp_probs
            normalized = {cls: prob / sum(probs.values()) for cls, prob in probs.items()}
            result.append(normalized)
        return result

