#
# create_balanced_dataset.py
# (فاز 2، بخش نهایی: ساخت دیتاست متعادل با Downsampling)
#
import os
import json
import random
import time
from tqdm import tqdm  # برای نوار پیشرفت

# --- تنظیمات ---

# فایلی که توسط preprocess_nlp.py ساخته شد
INPUT_FILE = "../jsonl_datasetdataset/labeled_traces_ALL.jsonl"
# فایل خروجی نهایی برای فاز 3 (آموزش LSTM)
OUTPUT_FILE = "../jsonl_dataset/labeled_traces_BALANCED.jsonl"

# تعداد کل خطوط برای نوار پیشرفت (شما ارائه دادید)
TOTAL_LINES = 48877355

# برای اطمینان از نتایج قابل تکرار
random.seed(42)


# -----------------

def main():
    start_time_total = time.time()

    if not os.path.exists(INPUT_FILE):
        print(f"❌ خطا: فایل ورودی {INPUT_FILE} یافت نشد.")
        print("لطفاً ابتدا اسکریپت preprocess_nlp.py را اجرا کنید.")
        return

    print(f"--- 🏁 شروع ساخت دیتاست متعادل از {INPUT_FILE} ---")
    print("این فرآیند به دلیل خواندن دیتاست 48 میلیونی، ممکن است کمی طول بکشد...")

    # ==================================================================
    #  خواندن فایل برای جمع‌آوری تروجان‌ها و شمارش داده‌های سالم
    # ==================================================================
    print(f"\n--- مرحله 1 از 2: در حال جمع‌آوری نمونه‌های تروجان (Label 1)... ---")

    trojan_traces = []  # تمام نمونه‌های تروجان در اینجا ذخیره می‌شوند (در RAM)
    total_normal_count = 0  # فقط تعداد نمونه‌های سالم را می‌شماریم

    start_time_pass1 = time.time()

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            # استفاده از tqdm برای نمایش پیشرفت در مرحله 1
            for line in tqdm(f, total=TOTAL_LINES, desc="📊 مرحله 1: شمارش", unit="L"):
                try:
                    item = json.loads(line)
                    if item['label'] == 1:
                        trojan_traces.append(item)
                    else:
                        total_normal_count += 1
                except (json.JSONDecodeError, KeyError):
                    tqdm.write(f"  ⚠️ هشدار: خط معیوب رد شد: {line.strip()}")

    except Exception as e:
        print(f"\n❌ خطا در حین خواندن مرحله 1: {e}")
        return

    end_time_pass1 = time.time()
    k_trojans = len(trojan_traces)  # تعداد کل نمونه‌های تروجان

    print(f"--- ✅ مرحله 1 کامل شد (در {end_time_pass1 - start_time_pass1:.2f} ثانیه) ---")
    print(f"  📈 {k_trojans:,} نمونه تروجان (Label 1) پیدا شد (در RAM ذخیره شدند).")
    print(f"  📉 {total_normal_count:,} نمونه سالم (Label 0) شمارش شد.")

    if k_trojans == 0:
        print("❌ خطا: هیچ نمونه تروجانی (Label 1) در دیتاست شما پیدا نشد. پردازش متوقف شد.")
        return

    if total_normal_count == 0:
        print("❌ خطا: هیچ نمونه سالمی (Label 0) در دیتاست شما پیدا نشد. پردازش متوقف شد.")
        return

    # ==================================================================
    #  Downsampling نمونه‌های سالم
    # ==================================================================
    print(f"\n--- مرحله 2 از 2: در حال نمونه‌برداری (Downsampling) {k_trojans:,} نمونه سالم... ---")

    # محاسبه نرخ نمونه‌برداری (Downsampling Rate)
    # این همان تکنیکی است که در مقاله به آن Downsampling می‌گویند
    sampling_rate = k_trojans / total_normal_count
    print(f"  نرخ نمونه‌برداری (Sampling Rate) برای Label 0: {sampling_rate:.6f}")

    normal_traces_selected = []  # نمونه‌های سالم انتخاب شده در اینجا ذخیره می‌شوند
    start_time_pass2 = time.time()

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            # استفاده از tqdm برای نمایش پیشرفت در مرحله 2
            for line in tqdm(f, total=TOTAL_LINES, desc="📊 مرحله 2: نمونه‌برداری", unit="L"):
                try:
                    # ما فقط به نمونه‌های سالم در این مرحله اهمیت می‌دهیم
                    if '"label": 0' not in line:
                        continue

                    # روش نمونه‌برداری تصادفی:
                    # به جای بارگذاری همه چیز، به هر نمونه سالم یک شانس کوچک می‌دهیم
                    if random.random() < sampling_rate:
                        item = json.loads(line)
                        normal_traces_selected.append(item)

                except (json.JSONDecodeError):
                    pass  # خطاها در مرحله 1 گزارش شده‌اند

    except Exception as e:
        print(f"\n❌ خطا در حین خواندن مرحله 2: {e}")
        return

    end_time_pass2 = time.time()
    print(f"--- ✅ مرحله 2 کامل شد (در {end_time_pass2 - start_time_pass2:.2f} ثانیه) ---")
    print(f"  📉 {len(normal_traces_selected):,} نمونه سالم (Label 0) به صورت تصادفی انتخاب شد.")

    # ==================================================================
    #  ادغام، مخلوط کردن و ذخیره
    # ==================================================================
    print(f"\n--- 💾 در حال ادغام، مخلوط کردن و ذخیره دیتاست نهایی... ---")

    # اکنون هر دو لیست به راحتی در 16 گیگابایت رم جا می‌شوند
    balanced_dataset = trojan_traces + normal_traces_selected
    random.shuffle(balanced_dataset)  # مخلوط کردن داده‌ها برای آموزش بهتر

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in tqdm(balanced_dataset, desc="💾 در حال ذخیره", unit=" traces"):
                f.write(json.dumps(item) + "\n")

    except Exception as e:
        print(f"\n❌ خطا در ذخیره فایل نهایی {OUTPUT_FILE}: {e}")
        return

    end_time_total = time.time()

    print("\n" + "=" * 50)
    print("🏁 دیتاست متعادل (Balanced Dataset) با موفقیت ساخته شد")
    print("=" * 50)
    print(f"⏱️ زمان کل پردازش: {(end_time_total - start_time_total) / 60:.2f} دقیقه")
    print(f"📊 خروجی نهایی: {OUTPUT_FILE}")
    print(f"  - کل ردیابی‌ها (Traces): {len(balanced_dataset):,}")
    print(f"  - ردیابی‌های تروجان (Label 1): {len(trojan_traces):,}")
    print(f"  - ردیابی‌های سالم (Label 0): {len(normal_traces_selected):,}")
    print("\n✅ شما اکنون آماده ورود به فاز 3 (آموزش مدل LSTM) هستید.")


if __name__ == "__main__":
    main()