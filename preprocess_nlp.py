#
# preprocess_nlp.py
# (فاز 2: جمع‌آوری تمام داده‌های فاز 1 برای آموزش NLP و DL)
#
import os
import glob
import json
import time
from typing import List, Dict, Any, TextIO
from tqdm import tqdm

# --- تنظیمات ---
DATASET_ROOTS = ["TRIT-TC", "TRIT-TS"]

# نام فایل‌های خروجی به .jsonl تغییر کرد (JSON Lines)
CORPUS_OUTPUT_FILE = "corpus_ALL.jsonl"
LABELED_DATA_OUTPUT_FILE = "labeled_traces_ALL.jsonl"


# -----------------

def find_all_trace_files(root_folders: List[str]) -> List[str]:
    """
    تمام فایل‌های _traces.json را در تمام زیرپوشه‌ها پیدا می‌کند.
    """
    all_files = []
    print(f"🔍 در حال جستجو برای فایل‌های _traces.json در {root_folders}...")
    for root in root_folders:
        if not os.path.isdir(root):
            print(f"  ⚠️ هشدار: پوشه '{root}' یافت نشد. از آن عبور می‌کنیم.")
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
    (بهینه‌سازی شده برای RAM)
    فایل‌های JSON را می‌خواند و داده‌ها را به صورت جریانی در فایل‌های باز می‌نویسد.
    تعداد کل ردیابی‌های پردازش شده را برمی‌گرداند.
    """
    total_traces_processed = 0

    # --- ✨ استفاده از tqdm برای نوار پیشرفت ✨ ---
    # desc="..." متنی است که کنار نوار پیشرفت نمایش داده می‌شود
    print(f"Processing {len(trace_files)} files...")
    for filepath in tqdm(trace_files, desc="📊 Processing JSON files", unit="file"):

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                traces_in_file = json.load(f)

            if not isinstance(traces_in_file, list):
                tqdm.write(f"  ⚠️ هشدار: فایل {filepath} فرمت لیست ندارد. عبور شد.")
                continue

            for item in traces_in_file:
                if 'trace' in item and 'label' in item and 'gate' in item:

                    # 1. نوشتن جمله در فایل corpus (فرمت jsonl)
                    # (فقط خود جمله، بدون indent برای صرفه‌جویی در فضا)
                    corpus_file_handle.write(json.dumps(item['trace']) + "\n")

                    # 2. نوشتن آبجکت کامل در فایل داده‌های برچسب‌دار (فرمت jsonl)
                    labeled_file_handle.write(json.dumps(item) + "\n")

                    total_traces_processed += 1
                else:
                    tqdm.write(f"  ⚠️ هشدار: آیتم معیوب در {filepath}. عبور شد.")

        except json.JSONDecodeError:
            tqdm.write(f"  ❌ خطا: فایل {filepath} خراب است (JSONDecodeError). عبور شد.")
        except Exception as e:
            tqdm.write(f"  ❌ خطا: خطای ناشناخته در {filepath}: {e}. عبور شد.")

    return total_traces_processed


def main():
    start_time = time.time()

    # 1. پیدا کردن تمام فایل‌های JSON
    all_json_files = find_all_trace_files(DATASET_ROOTS)

    if not all_json_files:
        print("❌ خطا: هیچ فایل _traces.json پیدا نشد. آیا اسکریپت‌های فاز 1 را اجرا کرده‌اید؟")
        return

    print(f"✅ {len(all_json_files)} فایل _traces.json پیدا شد.")

    # 2. ساخت دیتاست‌های جامع (به روش جریانی)
    try:
        # فایل‌های خروجی را باز می‌کنیم
        with open(CORPUS_OUTPUT_FILE, 'w', encoding='utf-8') as corpus_f, \
                open(LABELED_DATA_OUTPUT_FILE, 'w', encoding='utf-8') as labeled_f:

            print(f"💾 در حال نوشتن خروجی‌ها در {CORPUS_OUTPUT_FILE} و {LABELED_DATA_OUTPUT_FILE}...")

            total_traces = build_comprehensive_datasets_streaming(all_json_files, corpus_f, labeled_f)

    except IOError as e:
        print(f"❌ خطا: امکان نوشتن در فایل‌های خروجی وجود ندارد: {e}")
        return
    except Exception as e:
        print(f"❌ خطای ناشناخته در حین پردازش جریانی: {e}")
        return

    if total_traces == 0:
        print("❌ خطا: هیچ ردیابی (trace) معتبری برای پردازش پیدا نشد.")
        return

    print("✅ پردازش جریانی داده‌ها با موفقیت انجام شد.")

    # 3. گزارش نهایی
    end_time = time.time()
    print("\n" + "=" * 50)
    print("🏁 فاز 2 (Preprocess NLP) با موفقیت کامل شد")
    print("=" * 50)
    print(f"⏱️ زمان کل: {end_time - start_time:.2f} ثانیه")
    print(f"📂 تعداد کل فایل‌های JSON پردازش شده: {len(all_json_files)}")
    print(f"💬 تعداد کل ردیابی‌ها (جملات) جمع‌آوری شده: {total_traces}")
    print(f"RAM Usage: (بسیار کم، به لطف پردازش جریانی)")
    print(f"📊 خروجی‌ها:")
    print(f"  1. {CORPUS_OUTPUT_FILE} (برای آموزش Net2Vec)")
    print(f"  2. {LABELED_DATA_OUTPUT_FILE} (برای آموزش آشکارساز LSTM)")


if __name__ == "__main__":
    main()