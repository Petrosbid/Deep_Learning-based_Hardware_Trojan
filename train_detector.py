#
# train_detector.py
# (فاز 3: اسکریپت اصلی آموزش مدل LSTM)
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time

# وارد کردن کلاس‌های سفارشی
from dataset import TrojanDataset, LABELED_DATA_FILE, EMBEDDING_FILE
from model import TrojanLSTM

# --- تنظیمات ---
BATCH_SIZE = 32  #
LEARNING_RATE = 0.001
NUM_EPOCHS = 5  #
TRAIN_SPLIT = 0.8  # 80% برای آموزش، 20% برای اعتبارسنجی
OUTPUT_MODEL_FILE = "trojan_detector.pth"


# -----------------

def main():
    start_time = time.time()

    # 1. بررسی و تنظیم دستگاه (GPU یا CPU)
    # (از گرافیک 3050 Ti شما استفاده خواهد کرد)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🚀 در حال استفاده از دستگاه: {device} ---")
    if device.type == 'cuda':
        print(f"نام GPU: {torch.cuda.get_device_name(0)}")

    # 2. بارگذاری دیتاست (از dataset.py)
    try:
        full_dataset = TrojanDataset(LABELED_DATA_FILE, EMBEDDING_FILE)
    except FileNotFoundError as e:
        print(e)
        return

    # 3. تقسیم داده‌ها به آموزشی (Train) و اعتبارسنجی (Validation)
    total_size = len(full_dataset)
    train_size = int(total_size * TRAIN_SPLIT)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"\n--- 📊 تقسیم دیتاست ---")
    print(f"کل نمونه‌ها: {total_size:,}")
    print(f"نمونه‌های آموزشی (Train): {len(train_dataset):,}")
    print(f"نمونه‌های اعتبارسنجی (Validation): {len(val_dataset):,}")

    # 4. ساخت DataLoader ها
    # DataLoader داده‌ها را در دسته‌های (Batch) 32 تایی به GPU می‌فرستد
    # (ما دیگر به Upsampling/Downsampling نیازی نداریم چون دیتاست از قبل متعادل است)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 5. مقداردهی اولیه مدل، تابع هزینه و بهینه‌ساز
    model = TrojanLSTM().to(device)
    criterion = nn.CrossEntropyLoss()  #
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- 🏋️ شروع آموزش مدل LSTM ---")

    best_val_accuracy = 0.0  # برای ذخیره بهترین مدل

    # 6. حلقه آموزش
    for epoch in range(NUM_EPOCHS):
        start_epoch_time = time.time()

        # --- بخش آموزش ---
        model.train()  # مدل را در حالت آموزش قرار بده
        train_loss = 0.0
        train_corrects = 0

        # استفاده از tqdm برای نوار پیشرفت آموزش
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]"):
            traces = batch['trace_tensor'].to(device)
            labels = batch['label'].to(device)

            # 1. صفر کردن گرادیان‌ها
            optimizer.zero_grad()

            # 2. Forward pass
            outputs = model(traces)
            loss = criterion(outputs, labels)

            # 3. Backward pass و بهینه‌سازی
            loss.backward()
            optimizer.step()

            # محاسبه آمار
            train_loss += loss.item() * traces.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)

        # --- بخش اعتبارسنجی ---
        model.eval()  # مدل را در حالت ارزیابی قرار بده (Dropout غیرفعال می‌شود)
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():  # محاسبات گرادیان را خاموش کن
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]  "):
                traces = batch['trace_tensor'].to(device)
                labels = batch['label'].to(device)

                outputs = model(traces)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * traces.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        # --- چاپ نتایج دوره ---
        epoch_time = time.time() - start_epoch_time
        avg_train_loss = train_loss / len(train_dataset)
        avg_train_acc = train_corrects.double() / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = val_corrects.double() / len(val_dataset)

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} (Time: {epoch_time:.2f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} | Val   Acc: {avg_val_acc:.4f}")

        # ذخیره بهترین مدل (مدلی که بالاترین دقت اعتبارسنجی را دارد)
        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            torch.save(model.state_dict(), OUTPUT_MODEL_FILE)
            print(f"  ✨ مدل بهتر پیدا شد! در {OUTPUT_MODEL_FILE} ذخیره شد.")

    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("🏁 فاز 3 (آموزش) با موفقیت کامل شد")
    print("=" * 50)
    print(f"⏱️ زمان کل آموزش: {(total_time) / 60:.2f} دقیقه")
    print(f"🎯 بهترین دقت اعتبارسنجی: {best_val_accuracy:.4f}")
    print(f"💾 مدل نهایی در {OUTPUT_MODEL_FILE} ذخیره شد.")
    print("\n✅ شما اکنون آماده ورود به فاز 4 (ارزیابی نهایی و رأی‌گیری) هستید.")


if __name__ == "__main__":
    main()