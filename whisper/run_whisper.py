# pip install git+https://github.com/openai/whisper.git

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from abc import ABC
import whisper
import argparse
import time

class config:
    epochs = 5
    batch_size = 16
    train_dataset_size = 1000
    test_dataset_size = 100
    max_iters = 1


class DummyDataset:
    def __init__(self, size=1000, seed=100):
        import random
        random.seed(seed)

        self.data = {}
        for i in range(size):
            seq_len = random.randint(80, 160)
            self.data[i] = (
                torch.rand((80, 3000,), dtype=torch.float32),
                torch.ones(seq_len, dtype=torch.int64),
                torch.ones(seq_len, dtype=torch.int64)
            )

    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)


class WhisperModel(nn.Module):
    def __init__(self, model):
        super(WhisperModel, self).__init__()
        self.w_model = model

    def forward(self, input_ids: torch.Tensor, dec_input_ids: torch.Tensor):
        with torch.no_grad():
            audio_features = self.w_model.encoder(input_ids)

        pred = self.w_model.decoder(dec_input_ids, audio_features)
        return pred


def collect_fn(batch):
    # padding
    input_ids, dec_input_ids, labels = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    dec_input_ids = torch.nn.utils.rnn.pad_sequence(dec_input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return [input_ids, dec_input_ids, labels]


def get_dataloader():
    train_dataset = DummyDataset(size=config.train_dataset_size)
    test_dataset = DummyDataset(size=config.test_dataset_size)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collect_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collect_fn)

    return train_dataloader, test_dataloader


def train():
    train_dataloader, test_dataloader = get_dataloader()

    init_time = time.time_ns()
    whisper_model_name = "small"
    whisper_model = whisper.load_model(whisper_model_name)

    model = WhisperModel(whisper_model)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time_ns()
    iterations = 0
    for echo in range(config.epochs):
        for batch_idx, (input_ids, dec_input_ids, labels) in enumerate(train_dataloader):
            input_ids, dec_input_ids, labels = input_ids.to(device), dec_input_ids.to(device), labels.to(device)

            pred = model(input_ids, dec_input_ids)
            loss = loss_fn(pred.view(-1, pred.size(-1)), labels.view(-1))

            # loss.backward()
            # optimizer.zero_grad()
            # optimizer.step()

            # flush the stdout buffer
            import sys
            sys.stdout.flush()
            iterations += 1
            if iterations == config.max_iters:
                print(f"Iteration reaches {iterations}")
                end_time = time.time_ns()
                print(f"Time taken: {((end_time - start_time) / 1e9):.3f} seconds")
                print(f"All time taken (model load + model execution): {((end_time - init_time) / 1e9):.3f} seconds")
                save_path = "whisper_model.pth"
                torch.save(model.state_dict(), save_path)
                exit()
        print(f"Epoch {echo} loss {loss.item():.5f} ------------------------")


def test():
    _, test_dataloader = get_dataloader()

    init_time = time.time_ns()
    whisper_model_name = "small"
    whisper_model = whisper.load_model(whisper_model_name)

    model = WhisperModel(whisper_model)
    model = model.to(device)
    model.load_state_dict(torch.load("whisper_model.pth"))
    model.eval()

    start_time = time.time_ns()
    iterations = 0
    for batch_idx, (input_ids, dec_input_ids, labels) in enumerate(test_dataloader):
        input_ids, dec_input_ids, labels = input_ids.to(device), dec_input_ids.to(device), labels.to(device)

        pred = model(input_ids, dec_input_ids)

        # flush the stdout buffer
        import sys
        sys.stdout.flush()
        iterations += 1
        if iterations == config.max_iters:
            print(f"Iteration reaches {iterations}")
            end_time = time.time_ns()
            print(f"Time taken: {((end_time - start_time) / 1e9):.3f} seconds")
            print(f"All time taken (model load + model execution): {((end_time - init_time) / 1e9):.3f} seconds")
            exit()

if __name__ == "__main__":
    des = "Whisper"
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--train_dataset_size", type=int, default=config.train_dataset_size)
    parser.add_argument("--test_dataset_size", type=int, default=config.test_dataset_size)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_iters", type=int, default=config.max_iters, help="Maximum number of iterations")
    parser.add_argument("-t", type=str, default="train", help="Train or test mode")
    args = parser.parse_args()

    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.train_dataset_size = args.train_dataset_size
    config.test_dataset_size = args.test_dataset_size
    device = args.device
    config.max_iters = args.max_iters
    train_or_test = args.t
    print(f"Running whisper model with batch size {config.batch_size} and {train_or_test} mode")

    if train_or_test == "train":
        train()
    elif train_or_test == "test":
        test()
    else:
        raise ValueError(f"Invalid argument: {train_or_test}")

