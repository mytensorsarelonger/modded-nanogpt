"""
Write tokenized text into .bin shards in the llm.c / modded-nanogpt format.

Shard format:
  Offset 0-1023:   256 x int32 little-endian header
    header[0] = 20240520  (magic, asserted on load)
    header[1] = 1         (version, asserted on load)
    header[2] = num_tokens (asserted against file size)
    header[3:256] = zeros
  Offset 1024+:    num_tokens x uint16 little-endian

This matches _load_data_shard in train_gpt_simple.py.
"""

import struct
import json
import sys
import hashlib
from pathlib import Path
import numpy as np

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_SIZE = 1024  # 256 * 4 bytes

def write_shard(tokens: np.ndarray, path: Path, num_tokens: int | None = None) -> dict:
    """
    Write tokens (uint16 numpy array) to a .bin shard file.

    Returns a manifest entry dict with metadata.
    """
    if num_tokens is None:
        num_tokens = len(tokens)

    assert tokens.dtype == np.uint16, f"Expected uint16, got {tokens.dtype}"
    assert num_tokens <= len(tokens), f"num_tokens {num_tokens} > len(tokens) {len(tokens)}"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build header: 256 int32 values
    header = np.zeros(256, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = num_tokens

    with open(path, "wb") as f:
        f.write(header.tobytes())  # 1024 bytes
        f.write(tokens[:num_tokens].tobytes())  # num_tokens * 2 bytes

    file_size = path.stat().st_size
    expected_size = HEADER_SIZE + num_tokens * 2
    assert file_size == expected_size, f"File size {file_size} != expected {expected_size}"

    # Compute hash for manifest
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        sha256.update(f.read())

    return {
        "shard_path": str(path),
        "num_tokens": num_tokens,
        "file_size": file_size,
        "sha256": sha256.hexdigest(),
        "magic": SHARD_MAGIC,
        "version": SHARD_VERSION,
    }


def read_shard(path: Path) -> np.ndarray:
    """Read a .bin shard file and return the tokens as a uint16 array."""
    path = Path(path)
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_SIZE), dtype=np.int32)
        assert header[0] == SHARD_MAGIC, f"Magic mismatch: {header[0]}"
        assert header[1] == SHARD_VERSION, f"Version mismatch: {header[1]}"
        num_tokens = int(header[2])

        payload = f.read()
        tokens = np.frombuffer(payload, dtype=np.uint16)
        assert len(tokens) == num_tokens, f"Token count mismatch: {len(tokens)} != {num_tokens}"
        return tokens.copy()


def round_trip_test():
    """Write a shard, read it back, and verify equality."""
    print("Running round-trip test...")

    # Create test data
    test_tokens = np.random.randint(0, 65536, size=10000, dtype=np.uint16)
    test_path = Path("data/shards/_test_roundtrip.bin")

    # Write
    entry = write_shard(test_tokens, test_path)
    print(f"  Wrote {entry['num_tokens']} tokens to {test_path}")
    print(f"  File size: {entry['file_size']} bytes")
    print(f"  SHA256: {entry['sha256'][:16]}...")

    # Read back
    read_back = read_shard(test_path)
    assert len(read_back) == len(test_tokens), f"Length mismatch: {len(read_back)} != {len(test_tokens)}"
    assert np.array_equal(read_back, test_tokens), "Token data mismatch!"

    # Verify header fields
    with open(test_path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_SIZE), dtype=np.int32)
        assert header[0] == SHARD_MAGIC
        assert header[1] == SHARD_VERSION
        assert header[2] == 10000
        assert np.all(header[3:] == 0), "Header padding not zero"

    print("  PASSED: round-trip test")

    # Cleanup
    test_path.unlink()
    print("  Cleaned up test file")


if __name__ == "__main__":
    round_trip_test()
