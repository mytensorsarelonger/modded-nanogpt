"""
Train a BPE tokenizer from scratch on the Gutenberg corpus.

Phase 0 uses GPT-2's tokenizer (tiktoken) for throwaway runs.
This script trains a real tokenizer on our actual corpus mix.

For Phase 0, we use tiktoken's GPT-2 tokenizer (50257 vocab, padded to 50304)
as specified in the plan. This script is for Phase 1+ when the mix is settled.

For now, this script provides:
1. GPT-2 tokenizer wrapper (via tiktoken) for Phase 0
2. BPE training from scratch for Phase 1+
"""

import os
import sys
import json
import re
import collections
from pathlib import Path

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Warning: tiktoken not installed. Install with: pip install tiktoken")

# GPT-2 regex pattern (same as tiktoken)
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

# For environments without tiktoken, we use a simple BPE implementation
# This is a minimal BPE trainer suitable for Phase 0

def get_corpus_files(raw_dir: Path):
    """Get all text files from the Gutenberg raw directory."""
    return sorted(raw_dir.glob("*.txt"))

def tokenize_gpt2(text: str):
    """Simple GPT-2 style tokenization (regex split)."""
    return GPT2_PATTERN.findall(text)

def train_bpe(corpus_files, vocab_size=65536, verbose=True):
    """
    Train a BPE tokenizer from scratch.

    This is a simplified implementation. For production, use tokenizers (HuggingFace).
    """
    if verbose:
        print(f"Training BPE tokenizer on {len(corpus_files)} files...")

    # Step 1: Build word frequency table
    word_freq = collections.Counter()
    for fpath in corpus_files:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
            words = tokenize_gpt2(text)
            word_freq.update(words)

    if verbose:
        total_words = sum(word_freq.values())
        print(f"  Total words: {total_words}")
        print(f"  Unique words: {len(word_freq)}")

    # Step 2: Initialize vocabulary with characters
    # Each word is represented as a sequence of character tokens
    vocab = {}
    merges = []

    # Get all unique characters
    chars = set()
    for word in word_freq:
        chars.update(word)

    # Assign initial token IDs (reserve 0-255 for bytes, then chars)
    next_id = 256
    for ch in sorted(chars):
        vocab[ch] = next_id
        next_id += 1

    # Step 3: Represent each word as a list of character tokens
    word_tokens = {}
    for word, freq in word_freq.items():
        tokens = [vocab[ch] for ch in word]
        word_tokens[word] = (tokens, freq)

    # Step 4: Iteratively merge most frequent pairs
    current_vocab_size = len(vocab)
    target_merges = vocab_size - current_vocab_size

    if verbose:
        print(f"  Initial vocab size: {current_vocab_size}")
        print(f"  Target merges: {target_merges}")

    for merge_idx in range(target_merges):
        # Count pair frequencies
        pair_freq = collections.Counter()
        for word, (tokens, freq) in word_tokens.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freq[pair] += freq

        if not pair_freq:
            break

        # Find most frequent pair
        best_pair = pair_freq.most_common(1)[0][0]
        new_id = next_id
        next_id += 1

        # Create reverse vocab for decoding
        id_to_token = {v: k for k, v in vocab.items()}
        id_to_token[new_id] = id_to_token[best_pair[0]] + id_to_token[best_pair[1]]
        vocab[id_to_token[best_pair[0]] + id_to_token[best_pair[1]]] = new_id

        merges.append(best_pair)

        # Apply merge to all words
        for word in word_tokens:
            tokens, freq = word_tokens[word]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            word_tokens[word] = (new_tokens, freq)

        if verbose and (merge_idx + 1) % 10000 == 0:
            print(f"  Merge {merge_idx + 1}/{target_merges}, vocab size: {next_id}")

    if verbose:
        print(f"  Final vocab size: {next_id}")
        print(f"  Total merges: {len(merges)}")

    return vocab, merges, id_to_token


def save_tokenizer(vocab, merges, id_to_token, path: Path):
    """Save tokenizer to disk."""
    path.mkdir(parents=True, exist_ok=True)

    # Save vocab (token string -> id)
    vocab_path = path / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # Save merges
    merges_path = path / "merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        for pair in merges:
            id_to_token_str = id_to_token
            f.write(f"{id_to_token_str[pair[0]]} {id_to_token_str[pair[1]]}\n")

    # Save id_to_token mapping
    id_map_path = path / "id_to_token.json"
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_to_token.items()}, f, ensure_ascii=False, indent=2)

    print(f"Tokenizer saved to {path}")


def get_gpt2_tokenizer():
    """Get GPT-2 tokenizer via tiktoken (for Phase 0)."""
    if not HAS_TIKTOKEN:
        raise ImportError("tiktoken not installed. Run: pip install tiktoken")
    return tiktoken.get_encoding("gpt2")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=65536)
    parser.add_argument("--phase", choices=["0", "1"], default="0",
                        help="Phase 0: use GPT-2 tokenizer. Phase 1: train custom BPE")
    args = parser.parse_args()

    raw_dir = Path("data/gutenberg/raw")
    corpus_files = get_corpus_files(raw_dir)

    if not corpus_files:
        print("No corpus files found. Run download_gutenberg.py first.")
        sys.exit(1)

    if args.phase == "0":
        # Phase 0: use GPT-2 tokenizer
        enc = get_gpt2_tokenizer()
        print(f"GPT-2 tokenizer loaded. Vocab size: {enc.n_vocab}")
        print(f"  Padded to: 50304 (multiple of 128)")

        # Quick test
        test_text = "The ancient text spoke of things best left unspoken."
        tokens = enc.encode(test_text)
        print(f"  Test encoding: {tokens[:10]}...")
        print(f"  Decoded: {enc.decode(tokens)}")

    else:
        # Phase 1: train custom BPE
        vocab, merges, id_to_token = train_bpe(corpus_files, vocab_size=args.vocab_size)

        # Pad to multiple of 128 for tensor core alignment
        padded_size = ((len(vocab) + 127) // 128) * 128
        print(f"  Padding vocab from {len(vocab)} to {padded_size}")

        save_tokenizer(vocab, merges, id_to_token, Path("data/tokenizer"))


if __name__ == "__main__":
    main()
