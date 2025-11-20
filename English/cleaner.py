import os
import glob
import argparse
from tqdm import tqdm
import time

# --- Settings ---

# Root folders to be searched
TARGET_ROOTS = ["../Dataset/TRIT-TC", "../Dataset/TRIT-TS"]

# File patterns to be deleted
PATTERNS_TO_DELETE = [
    "*_traces.json",
    "*_pin_graph.gpickle"
]


# -----------------

def find_files_to_delete(root_folders, patterns):
    """
    Recursively finds all files matching the patterns in target folders.
    """
    files_to_delete = []
    print(f"🔍 Searching for files in {root_folders}...")

    for root in root_folders:
        if not os.path.isdir(root):
            print(f"  ⚠️ Warning: Root folder '{root}' not found. Skipping it.")
            continue

        for pattern in patterns:
            # Use glob to recursively search in all subfolders
            search_pattern = os.path.join(root, '**', pattern)

            # recursive=True tells glob to search in all subfolders (**)
            found_files = glob.glob(search_pattern, recursive=True)
            files_to_delete.extend(found_files)

    return files_to_delete


def main():
    # --- 1. Set up Argument Parser for safe execution ---
    parser = argparse.ArgumentParser(
        description="""Cleanup script to delete generated .json and .gpickle files.
                     Runs in 'Dry Run' mode by default."""
    )

    parser.add_argument(
        '--force',
        action='store_true',  # If this flag exists, set value to True
        help="Perform actual deletion. If not used, only lists files."
    )

    args = parser.parse_args()

    # --- 2. Find files ---
    start_time = time.time()
    files_found = find_files_to_delete(TARGET_ROOTS, PATTERNS_TO_DELETE)

    if not files_found:
        print("✅ Project is already clean. No files to delete found.")
        return

    print(f"Found {len(files_found)} files to process.")
    print("-" * 50)

    # --- 3. Perform deletion or dry run ---
    deleted_count = 0
    failed_count = 0

    if args.force:
        print("⚠️ Warning: '--force' mode activated. Permanently deleting files...")
        time.sleep(2)  # Brief pause to read warning

        for filepath in tqdm(files_found, desc="🔥 Deleting", unit="file"):
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                tqdm.write(f"  [Failed] Error deleting {filepath}: {e}")
                failed_count += 1
    else:
        print("INFO: Running 'Dry Run' mode. No files will be deleted.")
        print("To perform actual deletion, run the script with --force.")
        print("\n--- Files that would be deleted: ---")

        for filepath in files_found:
            print(f"  [Dry Run] {filepath}")

        deleted_count = len(files_found)

    # --- 4. Final report ---
    end_time = time.time()
    print("-" * 50)
    print(f"🏁 Operation completed in {end_time - start_time:.2f} seconds.")

    if args.force:
        print(f"  ✅ Files deleted: {deleted_count}")
        print(f"  ❌ Deletion errors: {failed_count}")
    else:
        print(f"  Total files that *would* be deleted: {deleted_count}")


if __name__ == "__main__":
    main()