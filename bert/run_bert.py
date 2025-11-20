# reference: https://github.com/codertimo/BERT-pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from bert_pytorch import BERT
import random
import argparse
import time

# Configuration class for training
class config:
    vocab_size = 30522  # Example vocab size (BERT's usual vocab size)
    hidden_size = 768
    n_layers = 12
    attn_heads = 12
    dropout = 0.1
    epochs = 5
    batch_size = 16
    seq_len = 512
    lr = 5e-5
    train_dataset_size = 1000
    test_dataset_size = 1000
    max_iters = 1

# Dummy Dataset for Random Data
class DummyDataset:
    def __init__(self, size=1000, seq_len=128, vocab_size=30522, seed=100):
        self.data = {}
        random.seed(seed)
        len_var = 1.5 
        seq_len = random.randint(int(config.seq_len / len_var), config.seq_len)
        for i in range(size):
            input_ids = torch.randint(1, vocab_size, (seq_len,))
            target_ids = torch.randn(config.hidden_size, seq_len)
            self.data[i] = (input_ids, target_ids)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

# Collate function to pad sequences
def collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids = nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=0)
    return input_ids, target_ids

# Training Function
def train_bert():

    # Set random seeds
    torch.manual_seed(100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = DummyDataset(size=config.train_dataset_size, seq_len=config.seq_len, vocab_size=config.vocab_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    init_time = time.time_ns()
    # Initialize model
    model = BERT(
        vocab_size=config.vocab_size,
        hidden=config.hidden_size,
        n_layers=config.n_layers,
        attn_heads=config.attn_heads,
        dropout=config.dropout
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Ignore padding index
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    start_time = time.time_ns()
    iterations = 0
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        # print(f"Epoch {epoch + 1}/{config.epochs}")

        for step, (input_ids, target_ids) in enumerate(dataloader):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            # Forward pass
            outputs = model(input_ids, segment_info=torch.zeros_like(input_ids))
            logits = outputs.transpose(1, 2)  # Adjust dimensions for CrossEntropyLoss

            loss = criterion(logits, target_ids)

            # Backward pass
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # total_loss += loss.item()

            # flush the stdout buffer
            import sys
            sys.stdout.flush()
            # Stop early if max iterations reached
            iterations += 1
            if iterations == config.max_iters:
                print(f"Iteration reaches {iterations}")
                end_time = time.time_ns()
                print(f"Time taken: {((end_time - start_time) / 1e9):.3f} seconds")
                print(f"All time taken (model load + model execution): {((end_time - init_time) / 1e9):.3f} seconds")
                save_path = "bert_model.pth"
                torch.save(model.state_dict(), save_path)
                exit()
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(dataloader):.4f}")

    print("Training complete!")

def test_bert():
    torch.manual_seed(100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test data
    dataset = DummyDataset(size=config.test_dataset_size, seq_len=config.seq_len, vocab_size=config.vocab_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    
    init_time = time.time_ns()
    # Load the trained model
    model = BERT(
        vocab_size=config.vocab_size,
        hidden=config.hidden_size,
        n_layers=config.n_layers,
        attn_heads=config.attn_heads,
        dropout=config.dropout
    ).to(device)

    # Load the trained model
    model.load_state_dict(torch.load("bert_model.pth"))
    model.eval()

    # Run inference
    iterations = 0
    start_time = time.time_ns()
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            outputs = model(input_ids, segment_info=torch.zeros_like(input_ids))
            logits = outputs.transpose(1, 2)
            
            # flush the stdout buffer
            import sys
            sys.stdout.flush()
            iterations += 1
            if iterations == config.max_iters:
                end_time = time.time_ns()
                print(f"Time taken: {((end_time - start_time) / 1e9):.3f} seconds")
                print(f"All time taken (model load + model execution): {((end_time - init_time) / 1e9):.3f} seconds")
                exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=config.seq_len, help="Sequence length for training")
    parser.add_argument("--max_iters", type=int, default=config.max_iters, help="Maximum number of iterations")
    parser.add_argument("-t", type=str, default="train", help="Train or test mode")
    args = parser.parse_args()

    train_or_test = args.t

    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.seq_len = args.seq_len
    config.max_iters = args.max_iters
    print(f"Running bert model with batch size {config.batch_size} and {train_or_test} mode")

    if train_or_test == "train":
        train_bert()
    elif train_or_test == "test":
        test_bert()
    else:
        raise ValueError(f"Invalid argument: {train_or_test}")
