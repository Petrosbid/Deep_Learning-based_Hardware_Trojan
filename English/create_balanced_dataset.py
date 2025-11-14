#
# create_balanced_dataset.py
# (Phase 2, final part: Create balanced dataset with Downsampling)
#
import os
import json
import random
import time
from tqdm import tqdm  # For progress bar

# --- Settings ---

# File created by preprocess_nlp.py
INPUT_FILE = "../jsonl_datasetdataset/labeled_traces_ALL.jsonl"
# Final output file for Phase 3 (LSTM training)
OUTPUT_FILE = "../jsonl_dataset/labeled_traces_BALANCED.jsonl"

# Total number of lines for progress bar (you provided this)
TOTAL_LINES = 48877355

# For reproducible results
random.seed(42)


# -----------------

def main():
    start_time_total = time.time()

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file {INPUT_FILE} not found.")
        print("Please run the preprocess_nlp.py script first.")
        return

    print(f"--- 🏁 Start building balanced dataset from {INPUT_FILE} ---")
    print("This process may take a while due to reading the 48 million dataset...")

    # ==================================================================
    # Step 1: Read file to collect trojans and count normal data
    # ==================================================================
    print(f"\n--- Step 1 of 2: Collecting trojan samples (Label 1)... ---")

    trojan_traces = []  # All trojan samples will be stored here (in RAM)
    total_normal_count = 0  # Just count normal samples

    start_time_pass1 = time.time()

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            # Using tqdm for progress display in step 1
            for line in tqdm(f, total=TOTAL_LINES, desc="📊 Step 1: Counting", unit="L"):
                try:
                    item = json.loads(line)
                    if item['label'] == 1:
                        trojan_traces.append(item)
                    else:
                        total_normal_count += 1
                except (json.JSONDecodeError, KeyError):
                    tqdm.write(f"  ⚠️ Warning: Invalid line skipped: {line.strip()}")

    except Exception as e:
        print(f"\n❌ Error while reading step 1: {e}")
        return

    end_time_pass1 = time.time()
    k_trojans = len(trojan_traces)  # Total number of trojan samples

    print(f"--- ✅ Step 1 completed (in {end_time_pass1 - start_time_pass1:.2f} seconds) ---")
    print(f"  📈 {k_trojans:,} trojan samples (Label 1) found (stored in RAM).")
    print(f"  📉 {total_normal_count:,} normal samples (Label 0) counted.")

    if k_trojans == 0:
        print("❌ Error: No trojan samples (Label 1) found in your dataset. Processing stopped.")
        return

    if total_normal_count == 0:
        print("❌ Error: No normal samples (Label 0) found in your dataset. Processing stopped.")
        return

    # ==================================================================
    # Step 2: Downsampling of normal samples
    # ==================================================================
    print(f"\n--- Step 2 of 2: Sampling (Downsampling) {k_trojans:,} normal samples... ---")

    # Calculate sampling rate (Downsampling Rate)
    # This is the same technique mentioned in the paper as Downsampling
    sampling_rate = k_trojans / total_normal_count
    print(f"  Sampling Rate for Label 0: {sampling_rate:.6f}")

    normal_traces_selected = []  # Selected normal samples will be stored here
    start_time_pass2 = time.time()

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            # Using tqdm for progress display in step 2
            for line in tqdm(f, total=TOTAL_LINES, desc="📊 Step 2: Sampling", unit="L"):
                try:
                    # We only care about normal samples in this step
                    if '"label": 0' not in line:
                        continue

                    # Random sampling method:
                    # Instead of loading everything, give each normal sample a small chance
                    if random.random() < sampling_rate:
                        item = json.loads(line)
                        normal_traces_selected.append(item)

                except (json.JSONDecodeError):
                    pass  # Errors were reported in step 1

    except Exception as e:
        print(f"\n❌ Error while reading step 2: {e}")
        return

    end_time_pass2 = time.time()
    print(f"--- ✅ Step 2 completed (in {end_time_pass2 - start_time_pass2:.2f} seconds) ---")
    print(f"  📉 {len(normal_traces_selected):,} normal samples (Label 0) randomly selected.")

    # ==================================================================
    # Final Step: Merge, shuffle, and save
    # ==================================================================
    print(f"\n--- 💾 Merging, shuffling, and saving final dataset... ---")

    # Now both lists easily fit in 16GB RAM
    balanced_dataset = trojan_traces + normal_traces_selected
    random.shuffle(balanced_dataset)  # Shuffle data for better training

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in tqdm(balanced_dataset, desc="💾 Saving", unit=" traces"):
                f.write(json.dumps(item) + "\n")

    except Exception as e:
        print(f"\n❌ Error saving final file {OUTPUT_FILE}: {e}")
        return

    end_time_total = time.time()

    print("\n" + "=" * 50)
    print("🏁 Balanced Dataset successfully created")
    print("=" * 50)
    print(f"⏱️ Total processing time: {(end_time_total - start_time_total) / 60:.2f} minutes")
    print(f"📊 Final output: {OUTPUT_FILE}")
    print(f"  - Total traces: {len(balanced_dataset):,}")
    print(f"  - Trojan traces (Label 1): {len(trojan_traces):,}")
    print(f"  - Normal traces (Label 0): {len(normal_traces_selected):,}")
    print("\n✅ You are now ready to proceed to Phase 3 (LSTM model training).")


if __name__ == "__main__":
    main()