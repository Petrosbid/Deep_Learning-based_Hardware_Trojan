import os
import glob
import argparse
from tqdm import tqdm
import time

# --- تنظیمات ---

# پوشه‌های ریشه‌ای که باید جستجو شوند
TARGET_ROOTS = ["../Dataset/TRIT-TC", "../Dataset/TRIT-TS"]

# الگوهای فایلی که باید حذف شوند
PATTERNS_TO_DELETE = [
    "*_traces.json",
    "*_pin_graph.gpickle"
]


# -----------------

def find_files_to_delete(root_folders, patterns):
    """
    به صورت بازگشتی تمام فایل‌های مطابق با الگوها را در پوشه‌های هدف پیدا می‌کند.
    """
    files_to_delete = []
    print(f"🔍 در حال جستجو برای فایل‌ها در {root_folders}...")

    for root in root_folders:
        if not os.path.isdir(root):
            print(f"  ⚠️ هشدار: پوشه ریشه '{root}' یافت نشد. از آن عبور می‌کنیم.")
            continue

        for pattern in patterns:
            # استفاده از glob برای جستجوی بازگشتی در تمام زیرپوشه‌ها
            search_pattern = os.path.join(root, '**', pattern)

            # recursive=True به glob می‌گوید که در تمام زیرپوشه‌ها (**) بگردد
            found_files = glob.glob(search_pattern, recursive=True)
            files_to_delete.extend(found_files)

    return files_to_delete


def main():
    # --- 1. تنظیم Argument Parser برای اجرای ایمن ---
    parser = argparse.ArgumentParser(
        description="""اسکریپت پاکسازی برای حذف فایل‌های .json و .gpickle ساخته شده.
                     به طور پیش‌فرض در حالت 'Dry Run' (آزمایشی) اجرا می‌شود."""
    )

    parser.add_argument(
        '--force',
        action='store_true',  # اگر این پرچم وجود داشته باشد، مقدار را True می‌گذارد
        help="اجرای واقعی حذف. در صورت عدم استفاده، فقط فایل‌ها لیست می‌شوند."
    )

    args = parser.parse_args()

    # --- 2. پیدا کردن فایل‌ها ---
    start_time = time.time()
    files_found = find_files_to_delete(TARGET_ROOTS, PATTERNS_TO_DELETE)

    if not files_found:
        print("✅ پروژه از قبل تمیز است. هیچ فایلی برای حذف پیدا نشد.")
        return

    print(f"Found {len(files_found)} فایل برای پردازش.")
    print("-" * 50)

    # --- 3. اجرای حذف یا اجرای آزمایشی ---
    deleted_count = 0
    failed_count = 0

    if args.force:
        print("⚠️ هشدار: حالت '--force' فعال شد. در حال حذف دائمی فایل‌ها...")
        time.sleep(2)  # یک مکث کوتاه برای خواندن هشدار

        for filepath in tqdm(files_found, desc="🔥 در حال حذف", unit="فایل"):
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                tqdm.write(f"  [Failed] خطا در حذف {filepath}: {e}")
                failed_count += 1
    else:
        print("INFO: در حال اجرای 'Dry Run' (آزمایشی). هیچ فایلی حذف نخواهد شد.")
        print("برای حذف واقعی، اسکریپت را با --force اجرا کنید.")
        print("\n--- فایل‌هایی که حذف خواهند شد: ---")

        for filepath in files_found:
            print(f"  [Dry Run] {filepath}")

        deleted_count = len(files_found)

    # --- 4. گزارش نهایی ---
    end_time = time.time()
    print("-" * 50)
    print(f"🏁 عملیات کامل شد در {end_time - start_time:.2f} ثانیه.")

    if args.force:
        print(f"  ✅ فایل‌های حذف شده: {deleted_count}")
        print(f"  ❌ خطا در حذف: {failed_count}")
    else:
        print(f"  Total files that *would* be deleted: {deleted_count}")


if __name__ == "__main__":
    main()