#
# train_detector_updated.py
# (فاز 3: اسکریپت اصلی آموزش - اصلاح نهایی با num_workers=0)
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
import json

# وارد کردن کلاس‌های سفارشی
from dataset import TrojanDataset, LABELED_DATA_FILE, EMBEDDING_FILE
from model import TrojanLSTM

# --- تنظیمات ---
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
TRAIN_SPLIT = 0.8
OUTPUT_MODEL_FILE = "../Model/trojan_detector_final.pth"

NUM_WORKERS = 1


# ----------------------------------------------------

def get_unique_circuits(data_file):
    """
    (این تابع دیگر استفاده نمی‌شود زیرا "load-all-in-RAM" سریع‌تر است،
     اما آن را برای مراجعات بعدی نگه می‌داریم)
    """
    print(f"--- 🔍 در حال خواندن {data_file} برای یافتن مدارهای منحصربه‌فرد...")
    circuits = set()
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="🔍 یافتن مدارها"):
            try:
                if line.strip():
                    circuits.add(json.loads(line)['circuit'])
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"✅ {len(circuits)} مدار منحصربه‌فرد پیدا شد.")
    return list(circuits)


def main():
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🚀 در حال استفاده از دستگاه: {device} ---")
    if device.type == 'cuda':
        print(f"نام GPU: {torch.cuda.get_device_name(0)}")

    try:
        full_dataset = TrojanDataset(LABELED_DATA_FILE, EMBEDDING_FILE)
    except FileNotFoundError as e:
        print(e)
        return

    total_size = len(full_dataset)
    train_size = int(total_size * TRAIN_SPLIT)
    val_size = total_size - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"\n--- 📊 تقسیم دیتاست ---")
    print(f"کل نمونه‌ها: {total_size:,}")
    print(f"نمونه‌های آموزشی (Train): {len(train_dataset):,}")
    print(f"نمونه‌های اعتبارسنجی (Validation): {len(val_dataset):,}")

    print(f"--- ❗ استفاده از num_workers={NUM_WORKERS} (برای جلوگیری از قفل شدن RAM) ---")
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS)  # 0

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS)  # 0

    model = TrojanLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- 🏋️ شروع آموزش مدل LSTM ---")

    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        start_epoch_time = time.time()

        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        ACCUM_STEPS = 4
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")):
            traces = batch['trace_tensor'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss = loss / ACCUM_STEPS

            loss.backward()

            if (i + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * traces.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data).item()

        model.eval()
        val_loss = 0.0
        val_corrects = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]  "):
                traces = batch['trace_tensor'].to(device)
                labels = batch['label'].to(device)

                outputs = model(traces)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * traces.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()

        # --- چاپ نتایج دوره ---
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / len(train_dataset)
        avg_train_acc = float(train_corrects) / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = float(val_corrects) / len(val_dataset)

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} (Time: {epoch_time:.2f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
        print(f"  Val   Loss: {avg_val_loss:.4f} | Val   Acc: {avg_val_acc:.4f}")

        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            torch.save(model.state_dict(), OUTPUT_MODEL_FILE)
            print(f"  ✨ مدل بهتر پیدا شد! در {OUTPUT_MODEL_FILE} ذخیره شد.")

    total_time = time.time() - start_time
    print(f"\n⏱️ زمان کل آموزش: {(total_time) / 60:.2f} دقیقه")
    print(f"💾 مدل نهایی در {OUTPUT_MODEL_FILE} ذخیره شد.")
    print("\n✅ شما اکنون آماده ارزیابی نهایی (evaluate.py) هستید.")


if __name__ == "__main__":
    main()