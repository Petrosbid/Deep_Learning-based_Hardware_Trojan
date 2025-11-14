#
# evaluate.py
# (نسخه نهایی با تقسیم‌بندی صحیح بر اساس مدار - بدون نشت داده)
#
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import json
import random

# وارد کردن کلاس‌های سفارشی
try:
    from dataset_upldated import TrojanDataset, LABELED_DATA_FILE, EMBEDDING_FILE
    from model import TrojanLSTM
except ImportError:
    print("❌ خطا: مطمئن شوید فایل‌های dataset_upldated.py و model.py در همین پوشه قرار دارند.")
    exit()

# --- تنظیمات ---
BATCH_SIZE = 64
TRAIN_SPLIT = 0.8  # باید با train_detector.py یکسان باشد
MODEL_FILE = "../Model/trojan_detector_final.pth"  # نام مدل جدید
NUM_WORKERS = 4
random.seed(42)  # استفاده از seed یکسان برای تضمین تقسیم‌بندی یکسان


def get_unique_circuits(data_file):
    """
    (کپی شده از train_detector)
    یک بار فایل .jsonl را می‌خواند تا لیست مدارهای منحصربه‌فرد را استخراج کند.
    """
    print(f"--- 🔍 در حال خواندن {data_file} برای یافتن مدارهای منحصربه‌فرد...")
    circuits = set()
    with open(data_file, 'r', encoding='utf-8') as f:
        # برای سرعت، ممکن است نیازی به خواندن کل فایل نباشد،
        # اما برای اطمینان کامل، کل فایل را می‌خوانیم.
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

    # --- ✨ 1. (جدید) بازسازی دقیق تقسیم‌بندی بر اساس مدار ---
    all_circuits = get_unique_circuits(LABELED_DATA_FILE)
    random.shuffle(all_circuits)  # seed=42 تضمین می‌کند که ترتیب یکسان است

    split_index = int(len(all_circuits) * TRAIN_SPLIT)
    # ما فقط به مدارهای تست (اعتبارسنجی) نیاز داریم
    val_circuit_names = set(all_circuits[split_index:])

    print(f"\n--- 📊 بارگذاری مجموعه اعتبارسنجی (Test Set) ---")
    print(f"تعداد کل مدارهای تست: {len(val_circuit_names)}")

    # --- 2. (جدید) ساخت دیتاست فقط برای مدارهای تست ---
    print("\n(بارگذاری دیتاست اعتبارسنجی...)")
    val_dataset = TrojanDataset(LABELED_DATA_FILE, EMBEDDING_FILE,
                                allowed_circuits_list=val_circuit_names)

    print(f"کل ردیابی‌های تست: {len(val_dataset):,}")

    # 3. ساخت DataLoader برای تست
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    # 4. بارگذاری مدل آموزش‌دیده (مدل جدید)
    try:
        model = TrojanLSTM().to(device)
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()
        print(f"✅ مدل از {MODEL_FILE} با موفقیت بارگذاری شد.")
    except FileNotFoundError:
        print(f"❌ خطا: فایل مدل {MODEL_FILE} یافت نشد.")
        return
    except Exception as e:
        print(f"❌ خطا در بارگذاری مدل: {e}")
        return

    print("\n--- 🔬 شروع ارزیابی (گام 10 و 11 مقاله) ---")

    # 5. اجرای مدل روی داده‌های تست
    gate_votes = defaultdict(lambda: {'true_label': 0, 'votes': []})

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="📊 ارزیابی ردیابی‌ها (Traces)"):
            traces = batch['trace_tensor'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            gates = batch['gate']

            valid_indices = labels != -1
            if not valid_indices.any():
                continue

            traces, labels, gates = traces[valid_indices], labels[valid_indices], [gates[i] for i in
                                                                                   valid_indices.nonzero(as_tuple=True)[
                                                                                       0]]

            outputs = model(traces)

            _, preds = torch.max(outputs, 1)
            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()

            # 6. جمع‌آوری آراء برای VOTER
            for i in range(len(gates)):
                gate_name = gates[i]
                gate_votes[gate_name]['votes'].append(preds_cpu[i])
                gate_votes[gate_name]['true_label'] = labels_cpu[i]

    print("✅ ارزیابی ردیابی‌ها کامل شد.")
    print(f"🗳️ {len(gate_votes)} گیت (Component) منحصربه‌فرد برای رأی‌گیری پیدا شد.")

    # 7. پیاده‌سازی VOTER (گام 12 مقاله)
    print("\n--- 🗳️ شروع رأی‌گیری (Voter - گام 12 مقاله) ---")

    y_true_component = []  # برچسب واقعی گیت‌ها
    y_pred_component = []  # پیش‌بینی نهایی گیت‌ها

    for gate_name, data in gate_votes.items():
        votes = data['votes']
        if not votes: continue

        num_ht_votes = sum(votes)
        num_normal_votes = len(votes) - num_ht_votes

        final_prediction = 0
        if num_ht_votes > num_normal_votes:
            final_prediction = 1
        elif num_ht_votes == num_normal_votes:
            final_prediction = 1

        y_true_component.append(data['true_label'])
        y_pred_component.append(final_prediction)

    print("✅ رأی‌گیری کامل شد.")

    # 8. محاسبه نتایج نهایی (سطح گیت)
    print("\n" + "=" * 50)
    print("🏁 نتایج نهایی (سطح گیت - ارزیابی صحیح)")
    print("=" * 50)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true_component, y_pred_component).ravel()

        print(f"  True Positives (TP) - تروجان‌های پیدا شده: {tp:,}")
        print(f"  False Negatives (FN) - تروجان‌های از دست رفته: {fn:,}")
        print(f"  True Negatives (TN) - سالم‌های درست: {tn:,}")
        print(f"  False Positives (FP) - هشدارهای غلط: {fp:,}")
        print("-" * 50)

        # محاسبه معیارها
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        print(f"  📊 معیارهای نهایی مدل شما (بدون نشت داده):")
        print(f"  - TPR (Recall / Accuracy): {tpr * 100:.2f}%")
        print(f"  - TNR (Specificity):     {tnr * 100:.2f}%")
        print(f"  - PPV (Precision):       {ppv * 100:.2f}%")
        print(f"  - NPV:                   {npv * 100:.2f}%")

        print("\n" + "-" * 50)
        print("  📖 نتایج مقاله (برای مقایسه):")
        print("  - Comb. (TC): 79.29% TPR, 99.97% TNR, 87.75% PPV, 99.94% NPV")
        print("  - Seq. (TS):  93.46% TPR, 99.99% TNR, 98.92% PPV, 99.92% NPV")

    except ValueError:
        print("❌ خطا: به نظر می‌رسد هیچ نمونه‌ای در مجموعه اعتبارسنجی وجود نداشت.")
    except Exception as e:
        print(f"❌ خطا در محاسبه نتایج: {e}")

    total_time = time.time() - start_time
    print(f"\n⏱️ زمان کل ارزیابی: {total_time:.2f} ثانیه")


if __name__ == "__main__":
    main()