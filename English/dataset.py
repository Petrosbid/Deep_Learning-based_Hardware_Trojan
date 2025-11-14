#
# dataset.py
# (Final version: "Load all in RAM" + "allowed_circuits_list" filter)
#
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
from tqdm import tqdm
import os
import io

# --- Settings ---
LABELED_DATA_FILE = "../jsonl_dataset/labeled_traces_BALANCED.jsonl"
EMBEDDING_FILE = "../Model/net2vec.vectors"
EMBEDDING_DIM = 100
LOGIC_LEVEL = 4
MAX_TRACE_LENGTH = (2 * LOGIC_LEVEL) - 1  # (7)


# -----------------

class TrojanDataset(Dataset):
    """
    (Final and complete version)
    1. Reads all data in __init__ and keeps it in RAM.
    2. Accepts allowed_circuits_list argument to filter data.
    """

    def __init__(self, data_file, embedding_file, allowed_circuits_list: set = None):
        """
        If allowed_circuits_list is specified,
        only traces related to those circuits will be loaded.
        """
        self.data_file = data_file

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"❌ Dataset file {data_file} not found.")
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"❌ Embedding file {embedding_file} not found.")

        print("--- 1. Loading Net2Vec dictionary (Embeddings)...")
        # --- Fix typo ---
        self.embeddings = KeyedVectors.load_word2vec_format(embedding_file)
        # -------------------------
        self.zero_vector = np.zeros(EMBEDDING_DIM).astype(np.float32)
        print("✅ ... Dictionary loaded.")

        print(f"--- 2. Reading and filtering {data_file} (in RAM)...")

        self.data = []  # All data will be loaded here

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="📊 Reading and filtering", unit="L"):
                try:
                    line_stripped = line.strip()
                    if line_stripped:
                        item = json.loads(line_stripped)

                        # --- ✨✨✨ Filtering logic (Prevent data leakage) ✨✨✨ ---
                        if allowed_circuits_list is not None:
                            # If the filter list exists, check if this item's circuit
                            # is in the allowed list
                            if item.get('circuit') in allowed_circuits_list:
                                self.data.append(item)
                        else:
                            # If no filter list exists, add everything
                            self.data.append(item)
                        # --- End of filtering logic ---

                except (json.JSONDecodeError, KeyError):
                    # Skip corrupted lines or those missing 'circuit' key
                    tqdm.write(f"⚠️ Invalid line ignored: {line_stripped[:50]}...")

        self.total_samples = len(self.data)
        if self.total_samples == 0:
            print("⚠️ Warning: 0 samples loaded after filtering. Are the circuit names correct?")
        else:
            print(f"✅ ... {self.total_samples:,} samples (filtered) loaded in RAM.")
        print("--- Ready for training ---")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Returns a single sample directly from RAM (very fast).
        """
        item = self.data[idx]
        trace_words = item['trace']
        label = item['label']
        gate = item['gate']

        trace_tensor_data = np.zeros((MAX_TRACE_LENGTH, EMBEDDING_DIM), dtype=np.float32)

        for i, word in enumerate(trace_words):
            if i >= MAX_TRACE_LENGTH:
                break
            if word in self.embeddings:
                trace_tensor_data[i] = self.embeddings[word]

        return {
            "trace_tensor": torch.tensor(trace_tensor_data, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "gate": gate
        }


# ... (main section for testing) ...
if __name__ == "__main__":
    print("--- 🧪 Start testing TrojanDataset (Final and corrected version) ---")
    try:
        # Test loading without filter
        print("\n--- Loading test (without filter) ---")
        dataset_full = TrojanDataset(LABELED_DATA_FILE, EMBEDDING_FILE)
        print(f"Total samples: {len(dataset_full):,}")

        # Test loading with filter (assuming a circuit named 'c2670_T001' exists)
        print("\n--- Loading test (with filter) ---")
        dataset_filtered = TrojanDataset(LABELED_DATA_FILE, EMBEDDING_FILE, allowed_circuits_list={'c2670_T001'})
        print(f"Filtered samples: {len(dataset_filtered):,}")

        sample = dataset_filtered[0]
        print("\n--- First sample (filtered) ---")
        print(f"Gate: {sample['gate']}")
        print(f"Label: {sample['label']}")
        print(f"Tensor Shape: {sample['trace_tensor'].shape}")

        print("\n✅ Dataset script (final) works correctly.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"❌ Unknown error during testing: {e}")