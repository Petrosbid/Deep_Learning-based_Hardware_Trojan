#
# process_originals.py
# (اسکریپت پردازش فایل‌های سالم در original_designs)
#
import os
import json
import glob
import time
import networkx as nx
import pickle  # <--- + وارد کردن کتابخانه pickle
from typing import Tuple

# وارد کردن توابع از فایل‌های مجزا
import netlist_parser
import phase1_graph_utils


def process_original_file(netlist_path: str) -> Tuple[bool, str]:
    """
    یک فایل .v سالم را پردازش کرده و فایل gpickle. و json. مربوطه‌اش را ذخیره می‌کند.
    """
    try:
        # --- 1. تعریف نام‌ها ---
        base_name = os.path.basename(netlist_path)
        circuit_name = base_name.replace('.v', '')
        dir_name = os.path.dirname(netlist_path)
        log_path = ""  # مسیر خالی یا نامعتبر
        json_output_file = os.path.join(dir_name, base_name.replace('.v', '_traces.json'))
        graph_output_file = os.path.join(dir_name, base_name.replace('.v', '_pin_graph.gpickle'))

        # --- 2. اجرای خط لوله پردازش ---

        # فاز 0: پارس کردن
        trojan_names = netlist_parser.parse_trojan_log(log_path)
        netlist = netlist_parser.parse_netlist(netlist_path, trojan_names)
        if netlist is None:
            return (False, "Parse Error: Netlist is empty or failed to parse")

        # فاز 1: تبدیل به گراف
        pin_graph = phase1_graph_utils.convert_to_pin_graph(netlist)

        # --- 3. (اصلاح شده) ذخیره گراف روی دیسک ---
        try:
            # - nx.write_gpickle(pin_graph, graph_output_file)
            with open(graph_output_file, 'wb') as f:  # <--- + باز کردن فایل در حالت نوشتن باینری
                pickle.dump(pin_graph, f)  # <--- + استفاده از pickle.dump
        except Exception as e:
            return (False, f"Graph Save Error: {e}")
        # -------------------------------------

        # فاز 1: (شبیه‌سازی شده) استخراج بلوک‌ها
        net_blocks = phase1_graph_utils.generate_netlist_blocks(pin_graph, logic_level=4)

        # فاز 1: (شبیه‌سازی شده) استخراج ردیابی‌ها
        all_traces_dict = phase1_graph_utils.extract_pcp_traces(net_blocks)

        # --- 4. آماده‌سازی داده‌های خروجی (همه با برچسب 0) ---
        labeled_trace_data = []
        for center_gate, traces in all_traces_dict.items():
            label = 0  # همیشه 0 چون تروجانی در کار نیست
            for trace in traces:
                labeled_trace_data.append({
                    'trace': trace,
                    'label': label,
                    'gate': center_gate,
                    'circuit': circuit_name
                })

    except Exception as e:
        return (False, f"Processing Error: {e}")

    # --- 5. ذخیره خروجی JSON ---
    try:
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(labeled_trace_data, f, indent=2)
    except Exception as e:
        return (False, f"JSON Save Error: {e}")

    return (True, f"Success (Saved {len(labeled_trace_data)} traces and 1 graph)")


# --- تابع main بدون تغییر باقی می‌ماند ---
def main():
    DATASET_ROOT = "../Dataset"
    target_folders = ["TRIT-TC/original_designs", "TRIT-TS/original_designs"]

    print("🚀 Processing Original Designs (Saves Graph + Traces)...")
    print(f"Ignoring .spf files, processing only .v files.")

    files_to_process = []
    for target in target_folders:
        path_pattern = os.path.join(DATASET_ROOT, target, "*.v")
        files_found = glob.glob(path_pattern)
        files_to_process.extend(files_found)

    if not files_to_process:
        print("Error: No .v files found in original_designs folders.")
        return

    total_files = len(files_to_process)
    print(f"Found {total_files} original .v files to process.")

    success_count = 0
    fail_count = 0
    failed_files_log = []
    start_time = time.time()

    for i, file_path in enumerate(files_to_process):
        rel_path = os.path.relpath(file_path)
        print(f"[{i + 1}/{total_files}] Processing: {rel_path}...", end=" ", flush=True)
        status, message = process_original_file(file_path)
        if status:
            print(f"✅ {message}")
            success_count += 1
        else:
            print(f"❌ FAILED. Reason: {message}")
            fail_count += 1
            failed_files_log.append((rel_path, message))

    end_time = time.time()

    print("\n" + "=" * 50)
    print("🏁 Original Designs Processing Finished")
    print("=" * 50)
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    print(f"Total Files Processed: {total_files}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {fail_count}")

    if fail_count > 0:
        print("Failed Files Report:")
        for file, reason in failed_files_log:
            print(f"  - {file}\n    Reason: {reason}")


if __name__ == "__main__":
    main()