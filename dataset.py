import itertools
import os
from typing import List, Tuple
import numpy as np
import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

def stream_tinystories(split: str = "train", n_samples: int = 500_000) -> List[str]:
    """
    Streams stories from roneneldan/TinyStories on Hugging Face without downloading full dataset.
    """
    print(f"Streaming {n_samples:,} stories from TinyStories ({split} split)...")
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    stories = [ex["text"].strip() for ex in itertools.islice(ds, n_samples)]
    print(f"Loaded {len(stories):,} stories.")
    return stories


def tokenize_and_cache(
    stories: List[str],
    tokenizer: ByteLevelBPETokenizer,
    eot_id: int,
    cache_path: str,
    chunk_size: int = 5000
) -> np.ndarray:
    """
    Tokenizes a list of stories in chunks and persists them as a uint16 numpy array.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached tokens from {cache_path}...")
        return np.load(cache_path)

    print(f"Tokenizing {len(stories):,} stories into {cache_path}...")
    parts = []
    for i in range(0, len(stories), chunk_size):
        encs = tokenizer.encode_batch(stories[i : i + chunk_size])
        parts.append(
            np.concatenate([np.array(e.ids + [eot_id], dtype=np.uint16) for e in encs])
        )
    tokens = np.concatenate(parts)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.save(cache_path, tokens)
    print(f"Saved {len(tokens):,} tokens to {cache_path}.")
    return tokens


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a random batch of input and target sequence pairs.
    """
    ix = np.random.randint(0, len(data) - block_size - 1, batch_size)
    x = torch.from_numpy(np.stack([data[i : i + block_size].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1 : i + block_size + 1].astype(np.int64) for i in ix]))
    return x.to(device), y.to(device)
