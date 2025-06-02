import os
# Disable malloc logging to suppress macOS warnings
os.environ["MallocStackLogging"] = "0"
import sys
import time
import torch
import numpy as np
from contextlib import contextmanager
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = [tokenizer(x,
                            max_length=100,
                            truncation=True,
                            padding='max_length',
                            return_tensors='pt') for x in X]
        self.Y = torch.tensor(np.array(Y).astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        encoded = self.X[index]
        inputs = {k: v.squeeze(0) for k, v in encoded.items()}
        if 'token_type_ids' not in inputs:
            inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
        return inputs, self.Y[index]

    def collate_fn(batch):
        inputs = {key: torch.stack([b[0][key] for b in batch]) for key in batch[0][0]}
        labels = torch.stack([b[1] for b in batch])
        return inputs, labels


@contextmanager
def suppress_stderr():
    # Context manager suppresses stderr temporarily.
    old_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


def benchmark_num_workers(dataset, batch_size=32, max_workers=8):
    cpu_count = os.cpu_count() or 4
    max_workers = min(max_workers, cpu_count)
    results = {}

    for nw in range(0, max_workers + 1):
        print(f"Testing num_workers={nw}")
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, pin_memory=True)

        start = time.time()
        with suppress_stderr():
            for _ in loader:
                pass
        duration = time.time() - start
        print(f"num_workers={nw}: loading 1 epoch took {duration:.2f} seconds")
        results[nw] = duration

    best_nw = min(results, key=results.get)
    print(f"\nBest num_workers: {best_nw} with {results[best_nw]:.2f} seconds")
    return best_nw
