import os
import json
import glob
import time
import pickle
from typing import Tuple

import netlist_parser
import phase1_graph_utils
FORCE_RERUN = True


# ------------------------------

def process_directory(dir_path: str) -> Tuple[bool, str]:
    try:
        folder_name = os.path.basename(os.path.normpath(dir_path))
        netlist_file = os.path.join(dir_path, f"{folder_name}.v")
        log_file = os.path.join(dir_path, "log.txt")
        json_output_file = os.path.join(dir_path, f"{folder_name}_traces.json")
        graph_output_file = os.path.join(dir_path, f"{folder_name}_pin_graph.gpickle")

        if not os.path.exists(netlist_file):
            return (False, "Read Error: .v file missing")
        if not os.path.exists(log_file):
            log_file = ""

            # --- اجرای خط لوله پردازش ---

        # فاز 0: پارس کردن
        trojan_names = netlist_parser.parse_trojan_log(log_file)
        netlist = netlist_parser.parse_netlist(netlist_file, trojan_names)
        if netlist is None:
            return (False, "Parse Error: Netlist is empty or failed to parse")

        pin_graph = phase1_graph_utils.convert_to_pin_graph(netlist)

        net_blocks = phase1_graph_utils.generate_netlist_blocks(pin_graph, logic_level=4)

        all_traces_dict = phase1_graph_utils.extract_pcp_traces(net_blocks)

        # ذخیره گراف
        try:
            with open(graph_output_file, 'wb') as f:
                pickle.dump(pin_graph, f)
        except Exception as e:
            return (False, f"Graph Save Error: {e}")

        # برچسب‌گذاری داده‌ها
        labeled_trace_data = []
        for center_gate, traces in all_traces_dict.items():
            label = 1 if center_gate in trojan_names else 0
            for trace in traces:
                labeled_trace_data.append({
                    'trace': trace,
                    'label': label,
                    'gate': center_gate,
                    'circuit': folder_name
                })

    except Exception as e:
        return (False, f"Processing Error: {e}")

    # ذخیره خروجی JSON
    try:
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(labeled_trace_data, f, indent=2)
    except Exception as e:
        return (False, f"JSON Save Error: {e}")

    return (True, f"Success (Saved {len(labeled_trace_data)} traces and 1 graph)")


def main():
    DATASET_ROOT = "../Dataset"
    target_folders = ["TRIT-TC", "TRIT-TS"]

    print(f"🚀 Batch Processing Started (Force Re-run: {FORCE_RERUN})...")

    folders_to_process = []
    for target in target_folders:
        path_pattern = os.path.join(DATASET_ROOT, target, "*/")
        all_subfolders = glob.glob(path_pattern)
        for folder in all_subfolders:
            if "original_designs" not in folder:
                folders_to_process.append(folder)

    total_folders = len(folders_to_process)
    print(f"Found {total_folders} Trojan circuit folders to process.")

    success_count = 0
    fail_count = 0
    skipped_count = 0
    failed_folders_log = []
    start_time = time.time()

    for i, folder_path in enumerate(folders_to_process):
        rel_path = os.path.relpath(folder_path)
        print(f"[{i + 1}/{total_folders}] Processing: {rel_path}...", end=" ", flush=True)
        status, message = process_directory(folder_path)
        if status:
            print(f"✅ {message}")
            if "Skipped" in message:
                skipped_count += 1
            else:
                success_count += 1
        else:
            print(f"❌ FAILED. Reason: {message}")
            fail_count += 1
            failed_folders_log.append((rel_path, message))

    end_time = time.time()

    print("\n" + "=" * 50)
    print("🏁 Batch Processing Finished")
    print("=" * 50)
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    print(f"Total Folders Checked: {total_folders}")
    print(f"✅ Newly Processed: {success_count}")
    print(f"⏩ Skipped (Already Done): {skipped_count}")
    print(f"❌ Failed: {fail_count}")

    if fail_count > 0:
        print("Failed Folders Report:")
        for folder, reason in failed_folders_log:
            print(f"  - {folder}\n    Reason: {reason}")


if __name__ == "__main__":
    main()