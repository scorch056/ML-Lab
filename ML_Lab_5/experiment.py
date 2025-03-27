import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

import numpy as np

NUM_CLIENTS = 2
BATCH_SIZE = 32


def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    
    # Split dataset: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    def normalize_images(batch):
        """Normalize images to [-1,1] range."""
        batch["img"] = [(np.array(img) / 127.5 - 1.0) for img in batch["img"]]
        return batch

    # Apply transformations (normalization)
    partition_train_test = partition_train_test.with_transform(normalize_images)
    
    def batch_generator(dataset, batch_size):
        """Simple NumPy-based batch generator."""
        data_x, data_y = np.array(dataset["img"]), np.array(dataset["label"])
        for i in range(0, len(data_x), batch_size):
            yield data_x[i : i + batch_size], data_y[i : i + batch_size]

    # Train and validation batch loaders
    trainloader = list(batch_generator(partition_train_test["train"], BATCH_SIZE))
    valloader = list(batch_generator(partition_train_test["test"], BATCH_SIZE))
    
    # Load global test set
    testset = fds.load_split("test").with_transform(normalize_images)
    testloader = list(batch_generator(testset, BATCH_SIZE))

    return trainloader, valloader, testloader

