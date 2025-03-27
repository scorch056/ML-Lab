import flwr as fl
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and preprocess data
def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    pca = PCA(n_components=100)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    
    return (x_train, y_train.ravel()), (x_test, y_test.ravel())

# Split data for clients
(train_x, train_y), (test_x, test_y) = load_data()
n_clients = 2
split_size = len(train_x) // n_clients
client_data = [
    (train_x[i * split_size: (i + 1) * split_size], train_y[i * split_size: (i + 1) * split_size])
    for i in range(n_clients)
]

# Define Flower client
class CIFARClient(fl.client.NumPyClient):
    def __init__(self, train_data, test_data):
        self.train_x, self.train_y = train_data
        self.test_x, self.test_y = test_data
        self.model = DecisionTreeClassifier()
    
    def get_parameters(self, config):
        return []  # DecisionTreeClassifier has no real numerical weights
    
    def set_parameters(self, parameters):
        pass  # No weights to update
    
    def fit(self, parameters, config):
        self.model.fit(self.train_x, self.train_y)
        return self.get_parameters(config), len(self.train_x), {}
    
    def evaluate(self, parameters, config):
        preds = self.model.predict(self.test_x)
        acc = accuracy_score(self.test_y, preds)
        return acc, len(self.test_x), {}

# Create a list of clients
def client_fn(cid):
    return CIFARClient(client_data[int(cid)], (test_x, test_y))

# Define FedAvg strategy
strategy = fl.server.strategy.FedAvg()

# Run Simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=n_clients,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
