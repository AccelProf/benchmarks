import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
import argparse
import random
import time
import sys


class config:
    epochs = 5
    batch_size = 32
    num_classes = 10
    train_dataset_size = 1000
    test_dataset_size = 1000
    max_iters = 1


class DummyDataset:
    def __init__(self, size=1000, seed=100):
        random.seed(seed)
        self.data = {}
        for i in range(size):
            input_tensor = torch.randn(3, 512, 512)
            # Generates a random int64 scalar between 1 and 10
            target = torch.rand(config.num_classes)
            self.data[i] = (input_tensor, target)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet18(num_classes, channels=3):
    return ResNet(Block, [2, 2, 2, 2], num_classes, channels)

def ResNet34(num_classes, channels=3):
    return ResNet(Block, [3, 4, 6, 3], num_classes, channels)

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)


def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)


def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)


def get_dataloader():
    training_data = DummyDataset(config.train_dataset_size)
    test_data = DummyDataset(config.test_dataset_size)

    train_dataloader = DataLoader(training_data, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size)

    return train_dataloader, test_dataloader


def train():
    train_dataloader, _ = get_dataloader()

    init_time = time.time_ns()
    if model_name == "resnet18":
        model = ResNet18(num_classes=config.num_classes)
    elif model_name == "resnet34":
        model = ResNet34(num_classes=config.num_classes)
    elif model_name == "resnet50":
        model = ResNet50(num_classes=config.num_classes)
    elif model_name == "resnet101":
        model = ResNet101(num_classes=config.num_classes)
    elif model_name == "resnet152":
        model = ResNet152(num_classes=config.num_classes)
    if model is None:
        raise ValueError("Invalid model name")
    model.to(device)
    optimizer = SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time_ns()
    iterations = 0
    for epoch in range(config.epochs):
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # flush the stdout buffer
            sys.stdout.flush()
            iterations += 1
            if iterations == config.max_iters:
                print(f"Iteration reaches {iterations}")
                end_time = time.time_ns()
                print(f"Time taken: {((end_time - start_time) / 1e9):.3f} seconds")
                print(f"All time taken (model load + model execution): {((end_time - init_time) / 1e9):.3f} seconds")
                torch.save(model.state_dict(), "resnet_model.pth")
                exit()

            if batch % 100 == 0 and batch > 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_dataloader.dataset)}]")


def inference():
    _, test_dataloader = get_dataloader()

    init_time = time.time_ns()
    if model_name == "resnet50":
        model = ResNet50(num_classes=config.num_classes)
    elif model_name == "resnet101":
        model = ResNet101(num_classes=config.num_classes)
    elif model_name == "resnet152":
        model = ResNet152(num_classes=config.num_classes)
    elif model_name == "resnet18":
        model = ResNet18(num_classes=config.num_classes)
    elif model_name == "resnet34":
        model = ResNet34(num_classes=config.num_classes)
    if model is None:
        raise ValueError("Invalid model name")
    
    # Load trained weights
    model.load_state_dict(torch.load("resnet_model.pth"))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    start_time = time.time_ns()
    iterations = 0
    
    # Run inference
    with torch.no_grad():  # Disable gradient computation
        for batch, (X, _) in enumerate(test_dataloader):
            X = X.to(device)
            pred = model(X)
            
            sys.stdout.flush()
            iterations += 1
            if iterations == config.max_iters:
                print(f"Iteration reaches {iterations}")
                end_time = time.time_ns()
                print(f"Time taken: {((end_time - start_time) / 1e9):.3f} seconds")
                print(f"All time taken (model load + model execution): {((end_time - init_time) / 1e9):.3f} seconds")
                exit()

            if batch % 100 == 0 and batch > 0:
                print(f"Processed batch {batch}")


def main(train_or_test):
    # Check if weights exist, if not train first
    try:
        if train_or_test == "train":
            train()
        elif train_or_test == "test":
            inference()
        else:
            raise ValueError(f"Invalid argument: {train_or_test}")
    except FileNotFoundError:
        print("Error: Please check the model name and the weights path!")



if __name__ == "__main__":
    des = "ResNet"
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument("--model", type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_classes", type=int, default=config.num_classes)
    parser.add_argument("--train_dataset_size", type=int, default=config.train_dataset_size)
    parser.add_argument("--test_dataset_size", type=int, default=config.test_dataset_size)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_iters", type=int, default=config.max_iters, help="Maximum number of iterations")
    parser.add_argument("-t", type=str, default="train", help="Train or test mode")
    args = parser.parse_args()

    model_name = args.model
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.num_classes = args.num_classes
    config.train_dataset_size = args.train_dataset_size
    config.test_dataset_size = args.test_dataset_size
    device = args.device
    config.max_iters = args.max_iters
    train_or_test = args.t

    print(f"Running {model_name} model with batch size {config.batch_size} and {train_or_test} mode")
    main(train_or_test)
