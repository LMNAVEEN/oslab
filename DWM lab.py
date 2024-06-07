#(2)DATA PREPROCESSING USING PYTHON-----------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

# Load dataset
dataset = pd.read_csv('Data.csv')

# Drop rows with missing values
dataset.dropna(inplace=True)

# Separate features and target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Label encoding for categorical feature (first column in this case)
unique_values = list(set(X[:, 0]))
label_map = {val: idx for idx, val in enumerate(unique_values)}
X[:, 0] = np.array([label_map[val] for val in X[:, 0]])

# One-hot encoding for the first column
num_unique = len(unique_values)
onehot_encoded = np.zeros((X.shape[0], num_unique))
for i, val in enumerate(X[:, 0]):
    onehot_encoded[i, int(val)] = 1

# Append the one-hot encoded columns to the rest of the data
X = np.hstack((onehot_encoded, X[:, 1:]))

# Label encoding for the target variable
unique_targets = list(set(y))
label_map_y = {val: idx for idx, val in enumerate(unique_targets)}
y = np.array([label_map_y[val] for val in y])

# Split the dataset into training and testing sets
test_size = 0.2
split_index = int(len(X) * (1 - test_size))
indices = np.arange(len(X))
np.random.shuffle(indices)
train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

# Feature scaling
mean_X_train = np.mean(X_train, axis=0)
std_X_train = np.std(X_train, axis=0)
X_train = (X_train - mean_X_train) / std_X_train
X_test = (X_test - mean_X_train) / std_X_train

# Print shapes of the resulting datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)



#(3)LINEAR REGRESSION USING PYTHON----------------------------------------------------------------------------------------------------
import random
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
random.seed(0)

# Generate random data for X and y
X = [2.5 * random.gauss(0, 1) + 1.5 for _ in range(100)]
res = [0.5 * random.gauss(0, 1) for _ in range(100)]
y = [2 + 0.3 * X[i] + res[i] for i in range(100)]

# Calculate the mean of X and y
xmean = sum(X) / len(X)
ymean = sum(y) / len(y)

# Calculate covariance of X and y and variance of X
xycov = [(X[i] - xmean) * (y[i] - ymean) for i in range(len(X))]
xvar = [(X[i] - xmean)**2 for i in range(len(X))]

# Calculate beta (slope) and alpha (intercept)
beta = sum(xycov) / sum(xvar)
alpha = ymean - (beta * xmean)

print(f'alpha = {alpha}')
print(f'beta = {beta}')

# Predict y values using the linear regression model
ypred = [alpha + beta * X[i] for i in range(len(X))]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(X, ypred, label='Predicted', color='blue')
plt.scatter(X, y, color='red', label='Actual')
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


#(4)IMPLEMENTATION OF APRIORI ALGORITHM-----------------------------------------------------
import csv
from collections import defaultdict

class Apriori:
    def __init__(self, min_support=0.01, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def fit(self, transactions):
        self.transactions = transactions
        self.itemsets = self._find_frequent_itemsets()
        self.rules = self._generate_rules()

    def _find_frequent_itemsets(self):
        itemsets = {}
        single_items = defaultdict(int)
        
        for transaction in self.transactions:
            for item in transaction:
                single_items[item] += 1
        
        single_items = {item: support for item, support in single_items.items() if support >= self.min_support * len(self.transactions)}
        itemsets[1] = single_items
        k = 2
        
        while len(itemsets.get(k - 1, [])) > 0:
            candidates = self._generate_candidates(itemsets[k - 1], k)
            itemsets[k] = self._filter_candidates(candidates, k)
            k += 1
        
        return itemsets

    def _generate_candidates(self, prev_itemsets, k):
        candidates = {}
        prev_itemsets = list(prev_itemsets.keys())
        
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                itemset1, itemset2 = sorted(prev_itemsets[i]), sorted(prev_itemsets[j])
                if itemset1[:k - 2] == itemset2[:k - 2]:
                    candidates[tuple(set(itemset1).union(itemset2))] = 0
        
        return candidates

    def _filter_candidates(self, candidates, k):
        itemsets = defaultdict(int)
        
        for transaction in self.transactions:
            for candidate in candidates:
                if set(candidate).issubset(transaction):
                    itemsets[candidate] += 1
        
        return {itemset: support for itemset, support in itemsets.items() if support >= self.min_support * len(self.transactions)}

    def _generate_rules(self):
        rules = []
        
        for itemset_size, itemsets in self.itemsets.items():
            if itemset_size < 2:
                continue
            
            for itemset in itemsets:
                for i in range(1, itemset_size):
                    antecedents = self._combinations(list(itemset), i)
                    
                    for antecedent in antecedents:
                        antecedent = tuple(sorted(antecedent))
                        consequent = tuple(sorted(set(itemset) - set(antecedent)))
                        
                        if antecedent in self.itemsets[len(antecedent)]:
                            confidence = itemsets[itemset] / self.itemsets[len(antecedent)][antecedent]
                            if confidence >= self.min_confidence:
                                rules.append((antecedent, consequent, confidence))
        
        return rules

    def _combinations(self, items, length):
        if length == 0:
            return [[]]
        combinations = []
        for i in range(len(items)):
            m = items[i]
            remaining = items[i + 1:]
            for p in self._combinations(remaining, length - 1):
                combinations.append([m] + p)
        return combinations

    def get_itemsets(self):
        return self.itemsets

    def get_rules(self):
        return self.rules

# Read transactions from the CSV file
transactions = []
with open(r'C:\Users\Dell\OneDrive\Desktop\Groceries_dataset.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        transactions.append(row)

# Remove header and empty items in transactions
transactions = transactions[1:]
transactions = [list(filter(None, transaction[1:])) for transaction in transactions]

# Apply the Apriori algorithm
apriori = Apriori(min_support=0.005, min_confidence=0.3)
apriori.fit(transactions)

# Create dataframes for itemsets and rules manually
itemsets_list = [(k, item, v) for k, items in apriori.get_itemsets().items() for item, v in items.items()]
rules_list = [(rule[0], rule[1], rule[2]) for rule in apriori.get_rules()]

print("Frequent Itemsets:")
for size, itemset, support in itemsets_list:
    print(f'Size: {size}, Itemset: {itemset}, Support: {support}')

print("\nAssociation Rules:")
for antecedent, consequent, confidence in rules_list:
    print(f'Antecedent: {antecedent}, Consequent: {consequent}, Confidence: {confidence}')


#(5)KNN ALGORITHM------------------------------------------------------------``
import random
import math

# Load the iris dataset manually
def load_iris():
    data = []
    with open('iris.data', 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                parts = line.strip().split(',')
                features = list(map(float, parts[:-1]))
                label = parts[-1]
                data.append(features + [label])
    
    # Encode labels manually
    label_map = {label: idx for idx, label in enumerate(sorted(set(row[-1] for row in data)))}
    for row in data:
        row[-1] = label_map[row[-1]]
    
    # Split into features and labels
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    
    return X, y

# Split dataset into training and testing sets manually
def train_test_split(X, y, test_size=0.3, random_state=42):
    random.seed(random_state)
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    split_index = int(len(X) * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

# Calculate Euclidean distance manually
def euclidean_distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

# Implement k-Nearest Neighbors classifier manually
class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = [(euclidean_distance(test_point, train_point), label) for train_point, label in zip(self.X_train, self.y_train)]
            distances.sort(key=lambda x: x[0])
            k_nearest_labels = [label for _, label in distances[:self.n_neighbors]]
            predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(predicted_label)
        return predictions

# Calculate accuracy manually
def accuracy_score(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

# Load dataset
X, y = load_iris()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



#(6)K MEANS CLUSTERING ALGORITHM------------------------------------------------------------------------------------------------------
import random
import math
import matplotlib.pyplot as plt

# Generate random blobs of data for clustering manually
def make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0):
    random.seed(random_state)
    X = []
    y = []
    for center_id in range(centers):
        center = [random.uniform(-10, 10) for _ in range(2)]
        for _ in range(n_samples // centers):
            point = [random.gauss(center[dim], cluster_std) for dim in range(2)]
            X.append(point)
            y.append(center_id)
    return X, y

# Calculate Euclidean distance manually
def euclidean_distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

# Implement K-Means clustering manually
class KMeans:
    def __init__(self, n_clusters=4, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = random.sample(X, self.n_clusters)
        
        for _ in range(self.max_iter):
            self.labels = [[] for _ in range(self.n_clusters)]
            
            # Assign points to the nearest centroid
            for point in X:
                distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
                closest_centroid = distances.index(min(distances))
                self.labels[closest_centroid].append(point)
            
            # Calculate new centroids
            new_centroids = []
            for label in self.labels:
                if label:  # avoid division by zero
                    new_centroid = [sum(dim) / len(label) for dim in zip(*label)]
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append(random.choice(X))
            
            # Check for convergence
            if new_centroids == self.centroids:
                break
            
            self.centroids = new_centroids

    def predict(self, X):
        predictions = []
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_centroid = distances.index(min(distances))
            predictions.append(closest_centroid)
        return predictions

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fit K-Means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Convert X to a more usable format for plotting
X_array = [list(item) for item in X]

# Plot the results
plt.scatter([x[0] for x in X_array], [x[1] for x in X_array], c=y_kmeans, s=50, cmap='viridis')
plt.scatter([centroid[0] for centroid in kmeans.centroids], [centroid[1] for centroid in kmeans.centroids], s=200, c='red', marker='X')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()



#(7)GAUSSIAN MIXTURE MODEL------------------------------------------------------------------------------
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Generate random blobs of data for clustering manually
def make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0):
    random.seed(random_state)
    X = []
    for center_id in range(centers):
        center = [random.uniform(-10, 10) for _ in range(2)]
        for _ in range(n_samples // centers):
            point = [random.gauss(center[dim], cluster_std) for dim in range(2)]
            X.append(point)
    return X

# Initialize random means, covariances, and weights for the GMM
def initialize_parameters(X, n_components):
    random.seed(0)
    n_features = len(X[0])
    means = random.sample(X, n_components)
    covariances = [np.identity(n_features) for _ in range(n_components)]
    weights = [1 / n_components for _ in range(n_components)]
    return means, covariances, weights

# Calculate the multivariate Gaussian probability density function
def gaussian_pdf(x, mean, covariance):
    size = len(x)
    det_cov = np.linalg.det(covariance)
    inv_cov = np.linalg.inv(covariance)
    norm_const = 1.0 / (math.pow((2 * np.pi), float(size) / 2) * math.pow(det_cov, 1.0 / 2))
    x_minus_mean = np.array(x) - np.array(mean)
    result = math.pow(math.e, -0.5 * (np.dot(np.dot(x_minus_mean, inv_cov), x_minus_mean.T)))
    return norm_const * result

# Expectation step: calculate responsibilities
def expectation(X, means, covariances, weights, n_components):
    responsibilities = []
    for x in X:
        resp = [weights[k] * gaussian_pdf(x, means[k], covariances[k]) for k in range(n_components)]
        total_resp = sum(resp)
        resp = [r / total_resp for r in resp]
        responsibilities.append(resp)
    return responsibilities

# Maximization step: update parameters
def maximization(X, responsibilities, n_components):
    n_samples = len(X)
    n_features = len(X[0])

    means = []
    covariances = []
    weights = []

    for k in range(n_components):
        resp_sum = sum([responsibilities[i][k] for i in range(n_samples)])
        new_mean = [sum([responsibilities[i][k] * X[i][dim] for i in range(n_samples)]) / resp_sum for dim in range(n_features)]
        means.append(new_mean)

        new_covariance = np.zeros((n_features, n_features))
        for i in range(n_samples):
            x_minus_mean = np.array(X[i]) - np.array(new_mean)
            new_covariance += responsibilities[i][k] * np.outer(x_minus_mean, x_minus_mean)
        new_covariance /= resp_sum
        covariances.append(new_covariance)

        new_weight = resp_sum / n_samples
        weights.append(new_weight)

    return means, covariances, weights

# Fit the GMM model to the data
def fit_gmm(X, n_components, max_iter=100):
    means, covariances, weights = initialize_parameters(X, n_components)
    
    for _ in range(max_iter):
        responsibilities = expectation(X, means, covariances, weights, n_components)
        means, covariances, weights = maximization(X, responsibilities, n_components)
    
    return means, covariances, weights, responsibilities

# Predict the labels for the data points
def predict_gmm(X, means, covariances, weights):
    responsibilities = expectation(X, means, covariances, weights, len(means))
    labels = [np.argmax(r) for r in responsibilities]
    return labels

# Generate data
X = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fit GMM model
means, covariances, weights, responsibilities = fit_gmm(X, n_components=4)

# Predict labels
labels = predict_gmm(X, means, covariances, weights)

# Convert X to a more usable format for plotting
X_array = np.array(X)

# Plot the results
plt.scatter(X_array[:, 0], X_array[:, 1], c=labels, s=50, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian Mixture Model Clustering')
plt.show()


#(8)OUTLIER DETECTION-----------------------------------------------------------------------------------------------------------
import random
import numpy as np
import matplotlib.pyplot as plt

# Generate random blobs of data for clustering manually
def make_blobs(n_samples=300, centers=1, cluster_std=0.3, random_state=42):
    random.seed(random_state)
    X = []
    for _ in range(n_samples):
        point = [random.gauss(0, cluster_std), random.gauss(0, cluster_std)]
        X.append(point)
    return X

# Generate uniform random outliers manually
def generate_outliers(n_outliers=50, low=-1, high=1):
    X_outliers = []
    for _ in range(n_outliers):
        point = [random.uniform(low, high), random.uniform(low, high)]
        X_outliers.append(point)
    return X_outliers

# Isolation Tree class
class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.root = None

    def fit(self, X, current_height=0):
        if current_height >= self.height_limit or len(X) <= 1:
            return {"size": len(X)}

        q = random.choice(range(len(X[0])))
        p = random.uniform(min([x[q] for x in X]), max([x[q] for x in X]))
        
        X_left = [x for x in X if x[q] < p]
        X_right = [x for x in X if x[q] >= p]

        return {
            "split_attribute": q,
            "split_value": p,
            "left": self.fit(X_left, current_height + 1),
            "right": self.fit(X_right, current_height + 1)
        }

    def path_length(self, x, node=None, current_height=0):
        if node is None:
            node = self.root

        if "size" in node:
            return current_height + self.c_factor(node["size"])

        q = node["split_attribute"]
        p = node["split_value"]

        if x[q] < p:
            return self.path_length(x, node["left"], current_height + 1)
        else:
            return self.path_length(x, node["right"], current_height + 1)

    def c_factor(self, size):
        if size <= 1:
            return 0
        return 2 * (np.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size)

# Isolation Forest class
class IsolationForest:
    def __init__(self, n_estimators=100, max_samples=256, contamination=0.1):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.trees = []
        self.threshold = None

    def fit(self, X):
        self.trees = []
        height_limit = np.ceil(np.log2(self.max_samples))
        for _ in range(self.n_estimators):
            X_sample = random.sample(X, min(len(X), self.max_samples))
            tree = IsolationTree(height_limit)
            tree.root = tree.fit(X_sample)
            self.trees.append(tree)

        scores = self.anomaly_score(X)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))

    def anomaly_score(self, X):
        scores = []
        for x in X:
            path_lengths = [tree.path_length(x) for tree in self.trees]
            score = 2 ** (-np.mean(path_lengths) / self.c_factor(len(X)))
            scores.append(score)
        return scores

    def predict(self, X):
        scores = self.anomaly_score(X)
        return [1 if score >= self.threshold else 0 for score in scores]

    def c_factor(self, size):
        if size <= 1:
            return 0
        return 2 * (np.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size)

# Generate data
X = make_blobs(n_samples=300, centers=1, cluster_std=0.3, random_state=42)
X_outliers = generate_outliers(n_outliers=50, low=-1, high=1)
X = X + X_outliers

# Fit Isolation Forest model
clf = IsolationForest(contamination=0.1)
clf.fit(X)

# Predict labels
y_pred = clf.predict(X)

# Convert X to a more usable format for plotting
X_array = np.array(X)

# Plot the results
plt.scatter(X_array[:, 0], X_array[:, 1], c=y_pred, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Outlier Detection using Isolation Forest')
plt.show()


#(====9====)CLASSIFICATION ALGORITHM- RANDOM FOREST CLASSIFIER---------------------------------------------------------------
import numpy as np
import random

# Load the Iris dataset manually
def load_iris():
    data = np.loadtxt('iris.data', delimiter=',', usecols=(0, 1, 2, 3))
    target = np.loadtxt('iris.data', delimiter=',', usecols=(4), dtype=str)
    target = np.array([0 if t == 'Iris-setosa' else 1 if t == 'Iris-versicolor' else 2 for t in target])
    return data, target

# Train-test split manually
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        random.seed(random_state)
    indices = list(range(len(X)))
    random.shuffle(indices)
    test_size = int(len(X) * test_size)
    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Random Forest implementation
class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            if self.random_state:
                np.random.seed(self.random_state + _)
            n_samples = len(X)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            tree = DecisionTreeClassifier(max_features=self.max_features)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(tree_preds, axis=0)).astype(int)

class DecisionTreeClassifier:
    def __init__(self, max_features='sqrt'):
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1:
            return y[0]
        if len(y) == 0:
            return None

        feature_indices = self._get_feature_indices(X.shape[1])
        best_feature, best_threshold = self._best_split(X, y, feature_indices)
        if best_feature is None:
            return np.bincount(y).argmax()

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold, 'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y, feature_indices):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                gini = self._gini(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _gini(self, left, right):
        left_size, right_size = len(left), len(right)
        total_size = left_size + right_size
        if total_size == 0:
            return 0
        gini_left = 1 - sum((np.bincount(left) / left_size) ** 2) if left_size > 0 else 0
        gini_right = 1 - sum((np.bincount(right) / right_size) ** 2) if right_size > 0 else 0
        return (left_size / total_size) * gini_left + (right_size / total_size) * gini_right

    def _get_feature_indices(self, n_features):
        if self.max_features == 'sqrt':
            return random.sample(range(n_features), int(np.sqrt(n_features)))
        return range(n_features)

    def _predict_one(self, x, tree):
        if isinstance(tree, dict):
            if x[tree['feature']] < tree['threshold']:
                return self._predict_one(x, tree['left'])
            else:
                return self._predict_one(x, tree['right'])
        else:
            return tree

# Accuracy calculation manually
def accuracy_score(y_true, y_pred):
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)

# Load dataset and preprocess
X, y = load_iris()
X_binary = X[y != 2]
y_binary = y[y != 2]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# Fit the Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
