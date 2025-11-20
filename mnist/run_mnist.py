import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse


class config:
    epochs = 5
    batch_size = 64
    num_classes = 10
    train_dataset_size = 2000
    test_dataset_size = 1000
    max_iters = 1


class DummyDataset:
    def __init__(self, size=1000):
        self.data = {}
        for i in range(size):
            input_tensor = torch.randn(1, 28, 28)
            # Generates a random int64 scalar between 1 and 10
            target = torch.randint(1, 10, (1,), dtype=torch.int64).item()
            self.data[i] = (input_tensor, target)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class MNISTNeuralNetwork(nn.Module):
    def __init__(self):
        super(MNISTNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10), # 10 classes
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def prepare_dataset():
    training_data = DummyDataset(size=config.train_dataset_size)
    test_data = DummyDataset(size=config.test_dataset_size)

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size)

    return train_dataloader, test_dataloader


iterations = 0
# Training the model
def train(dataloader, model, loss_fn, optimizer):
    global iterations
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iterations := iterations + 1) == config.max_iters:
            print(f"Iteration reaches {iterations}")
            exit()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


# Testing the model
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    train_dataloader, test_dataloader = prepare_dataset()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MNISTNeuralNetwork().to(device)
    # print(model)

    # Optimizing the model parameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Training and testing the model
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        # test(test_dataloader, model, loss_fn)

    print("Done!")


if __name__ == '__main__':
    des = "MNIST Neural Network"
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_classes", type=int, default=config.num_classes)
    parser.add_argument("--train_dataset_size", type=int, default=config.train_dataset_size)
    parser.add_argument("--test_dataset_size", type=int, default=config.test_dataset_size)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.num_classes = args.num_classes
    config.train_dataset_size = args.train_dataset_size
    config.test_dataset_size = args.test_dataset_size
    device = args.device
    
    main()