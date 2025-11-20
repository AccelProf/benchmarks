import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import sys


init_time = 0
start_time = 0
end_time = 0

class config:
    epochs = 5
    batch_size = 128
    num_classes = 10
    train_dataset_size = 2000
    test_dataset_size = 2000
    max_iters = 1


class DummyDataset:
    def __init__(self, size=1000):
        self.data = {}
        for i in range(size):
            input_tensor = torch.randn(3, 512, 512)
            target = torch.rand(config.num_classes)
            self.data[i] = (input_tensor, target)

    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)


class AlexNet(nn.Module):
    def __init__(self, num_classes, dropout: float = 0.5) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_dataloader():
    training_data = DummyDataset(config.train_dataset_size)
    test_data = DummyDataset(config.test_dataset_size)

    train_dataloader = DataLoader(training_data, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size)

    return train_dataloader, test_dataloader


def train():
    train_data, _ = get_dataloader()
    iterations = 0
    init_time = time.time_ns()
    model = AlexNet(config.num_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    size = len(train_data.dataset)
    model.train()

    start_time = time.time_ns()
    for t in range(config.epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        for batch, (X, y) in enumerate(train_data):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # flush the stdout buffer
            import sys
            sys.stdout.flush()
            iterations += 1
            if iterations == config.max_iters:
                print(f"Iteration reaches {iterations}")
                end_time = time.time_ns()
                print(f"Time taken: {((end_time - start_time) / 1e9):.3f} seconds")
                print(f"All time taken (model load + model execution): {((end_time - init_time) / 1e9):.3f} seconds")
                save_path = "weights.pth"
                torch.save(model.state_dict(), save_path)
                exit()

            if batch % 100 == 0 and batch > 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    save_path = "weights.pth"
    torch.save(model.state_dict(), save_path)

def inference():
    # Get test data
    _, test_data = get_dataloader()

    iterations = 0
    init_time = time.time_ns()
    # Load the trained model
    model = AlexNet(config.num_classes)
    model.load_state_dict(torch.load("weights.pth"))
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    start_time = time.time_ns()
    # Run inference
    with torch.no_grad():  # Disable gradient computation for inference
        for X, _ in test_data:
            X = X.to(device)
            output = model(X)
            # flush the stdout buffer
            import sys
            sys.stdout.flush()
            iterations += 1
            if iterations == config.max_iters:
                end_time = time.time_ns()
                print(f"Time taken: {((end_time - start_time) / 1e9):.3f} seconds")
                print(f"All time taken (model load + model execution): {((end_time - init_time) / 1e9):.3f} seconds")
                exit()

def main(train_or_test):
    if train_or_test == "train":
        train()
    elif train_or_test == "test":
        inference()
    else:
        raise ValueError(f"Invalid argument: {train_or_test}")


if __name__ == '__main__':
    des = "AlexNet"
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_classes", type=int, default=config.num_classes)
    parser.add_argument("--train_dataset_size", type=int, default=config.train_dataset_size)
    parser.add_argument("--test_dataset_size", type=int, default=config.test_dataset_size)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_iters", type=int, default=config.max_iters)
    # train or test
    parser.add_argument("-t", type=str, default="train")
    args = parser.parse_args()

    train_or_test = args.t

    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.num_classes = args.num_classes
    config.train_dataset_size = args.train_dataset_size
    config.test_dataset_size = args.test_dataset_size
    config.max_iters = args.max_iters
    device = args.device

    print(f"Running alexnet model with batch size {config.batch_size} and {train_or_test} mode")
    main(train_or_test)
