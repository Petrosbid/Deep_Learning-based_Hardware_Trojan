#
# preprocess_nlp.py
# (Phase 2: Collect all Phase 1 data for NLP and DL training)
#
import os
import glob
import json
import time
from typing import List, Dict, Any, TextIO
from tqdm import tqdm

# --- Settings ---
DATASET_ROOTS = ["../Dataset/TRIT-TC", "../Dataset/TRIT-TS"]

# Output file names changed to .jsonl (JSON Lines)
CORPUS_OUTPUT_FILE = "../jsonl_dataset/corpus_ALL.jsonl"
LABELED_DATA_OUTPUT_FILE = "../jsonl_dataset/labeled_traces_ALL.jsonl"


# -----------------

def find_all_trace_files(root_folders: List[str]) -> List[str]:
    """
    Finds all _traces.json files in all subdirectories.
    """
    all_files = []
    print(f"🔍 Searching for _traces.json files in {root_folders}...")
    for root in root_folders:
        if not os.path.isdir(root):
            print(f"  ⚠️ Warning: Folder '{root}' not found. Skipping it.")
            continue

        search_pattern = os.path.join(root, '**', '*_traces.json')
        files_found = glob.glob(search_pattern, recursive=True)
        all_files.extend(files_found)

    return all_files


def build_comprehensive_datasets_streaming(
        trace_files: List[str],
        corpus_file_handle: TextIO,
        labeled_file_handle: TextIO
) -> int:
    """
    (RAM-optimized)
    Reads JSON files and writes data as a stream to open files.
    Returns the total number of processed traces.
    """
    total_traces_processed = 0

    # --- ✨ Using tqdm for progress bar ✨ ---
    # desc="..." is the text displayed next to the progress bar
    print(f"Processing {len(trace_files)} files...")
    for filepath in tqdm(trace_files, desc="📊 Processing JSON files", unit="file"):

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                traces_in_file = json.load(f)

            if not isinstance(traces_in_file, list):
                tqdm.write(f"  ⚠️ Warning: File {filepath} does not have list format. Skipped.")
                continue

            for item in traces_in_file:
                if 'trace' in item and 'label' in item and 'gate' in item:

                    # 1. Write sentence to corpus file (jsonl format)
                    # (only the sentence itself, no indent to save space)
                    corpus_file_handle.write(json.dumps(item['trace']) + "\n")

                    # 2. Write full object to labeled data file (jsonl format)
                    labeled_file_handle.write(json.dumps(item) + "\n")

                    total_traces_processed += 1
                else:
                    tqdm.write(f"  ⚠️ Warning: Invalid item in {filepath}. Skipped.")

        except json.JSONDecodeError:
            tqdm.write(f"  ❌ Error: File {filepath} is corrupted (JSONDecodeError). Skipped.")
        except Exception as e:
            tqdm.write(f"  ❌ Error: Unknown error in {filepath}: {e}. Skipped.")

    return total_traces_processed


def main():
    start_time = time.time()

    # 1. Find all JSON files
    all_json_files = find_all_trace_files(DATASET_ROOTS)

    if not all_json_files:
        print("❌ Error: No _traces.json files found. Have you run Phase 1 scripts?")
        return

    print(f"✅ {len(all_json_files)} _traces.json files found.")

    # 2. Build comprehensive datasets (using streaming)
    try:
        # Open output files
        with open(CORPUS_OUTPUT_FILE, 'w', encoding='utf-8') as corpus_f, \
                open(LABELED_DATA_OUTPUT_FILE, 'w', encoding='utf-8') as labeled_f:

            print(f"💾 Writing outputs to {CORPUS_OUTPUT_FILE} and {LABELED_DATA_OUTPUT_FILE}...")

            total_traces = build_comprehensive_datasets_streaming(all_json_files, corpus_f, labeled_f)

    except IOError as e:
        print(f"❌ Error: Cannot write to output files: {e}")
        return
    except Exception as e:
        print(f"❌ Unknown error during streaming processing: {e}")
        return

    if total_traces == 0:
        print("❌ Error: No valid traces found for processing.")
        return

    print("✅ Data streaming processing completed successfully.")

    # 3. Final report
    end_time = time.time()
    print("\n" + "=" * 50)
    print("🏁 Phase 2 (Preprocess NLP) completed successfully")
    print("=" * 50)
    print(f"⏱️ Total time: {end_time - start_time:.2f} seconds")
    print(f"📂 Total JSON files processed: {len(all_json_files)}")
    print(f"💬 Total collected traces (sentences): {total_traces}")
    print(f"RAM Usage: (Very low, thanks to streaming processing)")
    print(f"📊 Outputs:")
    print(f"  1. {CORPUS_OUTPUT_FILE} (for Net2Vec training)")
    print(f"  2. {LABELED_DATA_OUTPUT_FILE} (for LSTM Detector training)")


if __name__ == "__main__":
    main()