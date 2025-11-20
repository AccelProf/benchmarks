import os
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
import time

from GPT2.model import GPT2LMHeadModel
from GPT2.config import GPT2Config
from GPT2.encoder import get_encoder


class config:
    epochs = 5
    batch_size = 8
    train_dataset_size = 1000
    test_dataset_size = 1000
    seq_len = 512
    lr = 5e-5
    max_iters = 1



# Dummy Dataset for Random Data
class DummyDataset:
    def __init__(self, size=1000, seed=100):
        import random
        random.seed(seed)

        self.data = {}
        for i in range(size):
            # seq_len = random.randint(80, 160)
            len_var = 1.5 
            seq_len = random.randint(int(config.seq_len / len_var), config.seq_len)
            self.data[i] = (
                torch.randint(0, 50257, (seq_len,)),  # Random token IDs
                torch.randint(0, 50257, (seq_len,))   # Target token IDs (same shape as input)
            )

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def collect_fn(batch):
    # padding
    input_ids, target_ids = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=0)
    return input_ids, target_ids


# Training Function
def train_gpt2(state_dict=None):
    # Set random seeds for reproducibility
    seed = 100
    np.random.seed(100)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dummy dataset
    dataset = DummyDataset(size=config.train_dataset_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collect_fn)

    init_time = time.time_ns()

    # Load GPT-2 Model
    enc = get_encoder()
    gpt2_config = GPT2Config()
    model = GPT2LMHeadModel(gpt2_config)

    # if state_dict:
    #     model.load_state_dict(state_dict)
    model.to(device)

    # Loss function and optimizer
    # loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index if any
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    start_time = time.time_ns()
    iterations = 0
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for step, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward pass
            loss = model(input_ids, lm_labels=target_ids)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # total_loss += loss.item()

            # flush the stdout buffer
            import sys
            sys.stdout.flush()
            iterations += 1
            if iterations == config.max_iters:
                print(f"Iteration reaches {iterations}")
                end_time = time.time_ns()
                print(f"Time taken: {((end_time - start_time) / 1e9):.3f} seconds")
                print(f"All time taken (model load + model execution): {((end_time - init_time) / 1e9):.3f} seconds")
                save_path = "gpt2_model.pth"
                torch.save(model.state_dict(), save_path)
                exit()


        print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(dataloader):.4f}")

    print("Training complete!")

def test_gpt2():
    # Set random seeds for reproducibility
    seed = 100
    np.random.seed(100)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dummy dataset
    dataset = DummyDataset(size=config.test_dataset_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collect_fn)

    init_time = time.time_ns()
    # Load GPT-2 Model
    enc = get_encoder()
    gpt2_config = GPT2Config()
    model = GPT2LMHeadModel(gpt2_config)
    model.to(device)

    # Load trained weights
    model.load_state_dict(torch.load("gpt2_model.pth"))
    model.eval()

    start_time = time.time_ns()
    iterations = 0
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for step, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Forward pass
            loss = model(input_ids, lm_labels=target_ids)

            # total_loss += loss.item()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=config.seq_len, help="Sequence length for training")
    parser.add_argument("--max_iters", type=int, default=config.max_iters, help="Maximum number of iterations")
    parser.add_argument("-t", type=str, default="train", help="Train or test mode")
    args = parser.parse_args()

    config.epoch = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.seq_len = args.seq_len
    config.max_iters = args.max_iters
    train_or_test = args.t
    print(f"Running gpt2 model with batch size {config.batch_size} and {train_or_test} mode")

    if train_or_test == "train":
        train_gpt2()
    elif train_or_test == "test":
        test_gpt2()
    else:
        raise ValueError(f"Invalid argument: {train_or_test}")
