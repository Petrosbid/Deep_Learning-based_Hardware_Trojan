#
# dataset.py
# (نسخه نهایی: "Load all in RAM" + فیلتر "allowed_circuits_list")
#
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
from tqdm import tqdm
import os
import io

# --- تنظیمات ---
LABELED_DATA_FILE = "../jsonl_dataset/labeled_traces_BALANCED.jsonl"
EMBEDDING_FILE = "../Model/net2vec.vectors"
EMBEDDING_DIM = 100
LOGIC_LEVEL = 4
MAX_TRACE_LENGTH = (2 * LOGIC_LEVEL) - 1  # (7)


# -----------------

class TrojanDataset(Dataset):
    """
    (نسخه نهایی و کامل)
    1. تمام داده‌ها را در __init__ می‌خواند و در RAM نگه می‌دارد.
    2. آرگومان allowed_circuits_list را برای فیلتر کردن داده‌ها می‌پذیرد.
    """

    def __init__(self, data_file, embedding_file, allowed_circuits_list: set = None):
        """
        اگر allowed_circuits_list مشخص شده باشد،
        فقط ردیابی‌های مربوط به آن مدارها بارگذاری می‌شوند.
        """
        self.data_file = data_file

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"❌ فایل دیتاست {data_file} یافت نشد.")
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"❌ فایل Embedding {embedding_file} یافت نشد.")

        print("--- 1. در حال بارگذاری دیکشنری Net2Vec (Embeddings)...")
        # --- اصلاح اشتباه تایپی ---
        self.embeddings = KeyedVectors.load_word2vec_format(embedding_file)
        # -------------------------
        self.zero_vector = np.zeros(EMBEDDING_DIM).astype(np.float32)
        print("✅ ... دیکشنری بارگذاری شد.")

        print(f"--- 2. در حال خواندن و فیلتر کردن {data_file} (در RAM)...")

        self.data = []  # تمام داده‌ها در اینجا بارگذاری می‌شوند

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="📊 خواندن و فیلتر کردن", unit="L"):
                try:
                    line_stripped = line.strip()
                    if line_stripped:
                        item = json.loads(line_stripped)

                        # --- ✨✨✨ منطق فیلتر کردن (جلوگیری از نشت داده) ✨✨✨ ---
                        if allowed_circuits_list is not None:
                            # اگر لیست فیلتر وجود دارد، بررسی کن که آیا مدار این آیتم
                            # در لیست مجاز هست یا نه
                            if item.get('circuit') in allowed_circuits_list:
                                self.data.append(item)
                        else:
                            # اگر لیست فیلتری وجود ندارد، همه را اضافه کن
                            self.data.append(item)
                        # --- پایان منطق فیلتر ---

                except (json.JSONDecodeError, KeyError):
                    # از خطوط خراب یا فاقد کلید 'circuit' رد شو
                    tqdm.write(f"⚠️ خط خراب نادیده گرفته شد: {line_stripped[:50]}...")

        self.total_samples = len(self.data)
        if self.total_samples == 0:
            print("⚠️ هشدار: 0 نمونه پس از فیلتر کردن بارگذاری شد. آیا نام مدارها درست است؟")
        else:
            print(f"✅ ... {self.total_samples:,} نمونه (فیلتر شده) در RAM بارگذاری شد.")
        print("--- آماده برای آموزش ---")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        یک نمونه واحد را مستقیماً از RAM برمی‌گرداند (بسیار سریع).
        """
        item = self.data[idx]
        trace_words = item['trace']
        label = item['label']
        gate = item['gate']

        trace_tensor_data = np.zeros((MAX_TRACE_LENGTH, EMBEDDING_DIM), dtype=np.float32)

        for i, word in enumerate(trace_words):
            if i >= MAX_TRACE_LENGTH:
                break
            if word in self.embeddings:
                trace_tensor_data[i] = self.embeddings[word]

        return {
            "trace_tensor": torch.tensor(trace_tensor_data, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "gate": gate
        }


# ... (بخش __main__ برای تست) ...
if __name__ == "__main__":
    print("--- 🧪 شروع تست TrojanDataset (نسخه نهایی و اصلاح شده) ---")
    try:
        # تست بارگذاری بدون فیلتر
        print("\n--- تست بارگذاری (بدون فیلتر) ---")
        dataset_full = TrojanDataset(LABELED_DATA_FILE, EMBEDDING_FILE)
        print(f"تعداد کل نمونه‌ها: {len(dataset_full):,}")

        # تست بارگذاری با فیلتر (فرض می‌کنیم مداری به نام 'c2670_T001' وجود دارد)
        print("\n--- تست بارگذاری (با فیلتر) ---")
        dataset_filtered = TrojanDataset(LABELED_DATA_FILE, EMBEDDING_FILE, allowed_circuits_list={'c2670_T001'})
        print(f"نمونه‌های فیلتر شده: {len(dataset_filtered):,}")

        sample = dataset_filtered[0]
        print("\n--- نمونه اول (فیلتر شده) ---")
        print(f"Gate: {sample['gate']}")
        print(f"Label: {sample['label']}")
        print(f"Tensor Shape: {sample['trace_tensor'].shape}")

        print("\n✅ اسکریپت dataset (نهایی) به درستی کار می‌کند.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"❌ خطای ناشناخته در حین تست: {e}")