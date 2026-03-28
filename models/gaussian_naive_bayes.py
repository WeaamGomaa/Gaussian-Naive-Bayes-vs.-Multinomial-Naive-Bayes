import numpy as np
import utils


class GaussianNaiveBayes:
    def __init__(self):
        self.means = None
        self.vars = None
        self.priors = None
        self.classes = None

    def fit(self, X, y):
        # Convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Extract classes
        self.classes = np.unique(y)
        print(self.classes)

        # Compute class priors
        self.priors = {}
        for cls in self.classes:
            self.priors[cls] = np.sum(y == cls) / len(y)
        print(self.priors)

        # Compute mean and variance per feature per class
        self.means = {}
        self.vars = {}
        for cls in self.classes:
            X_cls = X[y == cls]  # Filter data for the current class
            self.means[cls] = [utils.compute_mean(X_cls[:, feature]) for feature in range(X_cls.shape[1])]
            self.vars[cls] = [utils.compute_var(X_cls[:, feature]) for feature in range(X_cls.shape[1])]
        print(self.means)
        print(self.vars)

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

        for sample in X:
            class_probs = {}
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
                    print("Feature value: ", feature_value)
                    print("Mean: ", mean)
                    print("Variance: ", var)

                    if use_log:
                        total_prob += self.calculate_log_likelihood(feature_value, mean, var)
                    else:
                        total_prob *= self.calculate_likelihood(feature_value, mean, var)
                    print("Total probability: ", total_prob)

                class_probs[cls] = total_prob

            # Get the class with the highest probability
            best_class = max(class_probs, key=class_probs.get)
            predictions.append(best_class)
            print(f"The class with the highest probability is: {best_class}")

        return predictions

# Dummy data
X = [[1, 2], [3, 4], [5, 6]]
y = ['A', 'B', 'A']
X_test = [[7, 8]]
y_test = ['A']
model = GaussianNaiveBayes()
model.fit(X, y)
model.predict(X_test)
