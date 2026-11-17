import os
from typing import Iterable, Optional
from tokenizers import ByteLevelBPETokenizer

EOT_TOKEN = "<|endoftext|>"

def train_bpe_tokenizer(
    text_iterator: Iterable[str],
    vocab_size: int = 4096,
    min_frequency: int = 2,
    save_dir: Optional[str] = None
) -> ByteLevelBPETokenizer:
    """
    Trains a Byte-Level Byte-Pair Encoding (BPE) tokenizer on the provided texts.
    """
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        text_iterator,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[EOT_TOKEN]
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_model(save_dir)
        print(f"Tokenizer saved to {save_dir}")

    return tokenizer


def load_bpe_tokenizer(tokenizer_dir: str) -> ByteLevelBPETokenizer:
    """
    Loads a trained ByteLevelBPETokenizer from a directory containing vocab.json and merges.txt.
    """
    vocab_file = os.path.join(tokenizer_dir, "vocab.json")
    merges_file = os.path.join(tokenizer_dir, "merges.txt")
    if not (os.path.exists(vocab_file) and os.path.exists(merges_file)):
        raise FileNotFoundError(f"Missing vocab.json or merges.txt in {tokenizer_dir}")

    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    return tokenizer
