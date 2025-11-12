#
# dataset.py
# (فاز 3: ابزار بارگذاری داده برای PyTorch)
#
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
from tqdm import tqdm
import os

# --- تنظیمات ---
LABELED_DATA_FILE = "labeled_traces_BALANCED.jsonl"
EMBEDDING_FILE = "net2vec.vectors"
EMBEDDING_DIM = 100  # باید با فایل net2vec.vectors مطابقت داشته باشد
LOGIC_LEVEL = 4
# طول ردیابی (trace) بر اساس مقاله 2*ll-1 است
MAX_TRACE_LENGTH = (2 * LOGIC_LEVEL) - 1  # (2*4-1 = 7)


# -----------------

class TrojanDataset(Dataset):
    """
    یک کلاس Dataset سفارشی PyTorch برای خواندن دیتاست متعادل.
    این کلاس داده‌ها را از دیسک می‌خواند و آن‌ها را به بردار تبدیل می‌کند.
    """

    def __init__(self, data_file, embedding_file):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"❌ فایل دیتاست {data_file} یافت نشد.")
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"❌ فایل Embedding {embedding_file} یافت نشد.")

        print("--- 1. در حال بارگذاری دیکشنری Net2Vec (Embeddings)...")
        # 1. بارگذاری دیکشنری بردارها
        # (این فایل کوچک است و به راحتی در رم جا می‌شود)
        self.embeddings = KeyedVectors.load_word2vec_format(embedding_file)
        # ایجاد یک بردار صفر برای کلماتی که در دیکشنری نیستند (padding)
        self.zero_vector = np.zeros(EMBEDDING_DIM).astype(np.float32)
        print("✅ ... دیکشنری بارگذاری شد.")

        print(f"--- 2. در حال خواندن دیتاست متعادل {data_file}...")
        # 2. خواندن تمام داده‌های برچسب‌دار
        # (فایل متعادل به اندازه کافی کوچک است که در رم 16 گیگابایتی جا شود)
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="📊 خواندن دیتاست متعادل"):
                try:
                    self.data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # نادیده گرفتن خطوط خراب احتمالی

        print(f"✅ ... {len(self.data):,} نمونه در حافظه بارگذاری شد.")
        print("--- آماده برای آموزش ---")

    def __len__(self):
        """تعداد کل نمونه‌ها در دیتاست را برمی‌گرداند."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        یک نمونه واحد از دیتاست را بر اساس ایندکس (idx) برمی‌گرداند.
        """
        # 1. آیتم را از دیتاست بارگذاری شده در رم بردار
        item = self.data[idx]
        trace_words = item['trace']
        label = item['label']
        gate = item['gate']  # برای فاز 4 (رای‌گیری) نیاز می‌شود

        # 2. ایجاد یک ماتریس صفر برای این ردیابی
        # شکل ماتریس: (7, 100) -> 7 کلمه، هر کدام 100 بعد
        trace_tensor_data = np.zeros((MAX_TRACE_LENGTH, EMBEDDING_DIM), dtype=np.float32)

        # 3. تبدیل کلمات به بردار (Vectorization)
        for i, word in enumerate(trace_words):
            if i >= MAX_TRACE_LENGTH:
                break  # ردیابی طولانی‌تر از حد مجاز را قطع کن

            # اگر کلمه در دیکشنری بود، بردار آن را قرار بده
            if word in self.embeddings:
                trace_tensor_data[i] = self.embeddings[word]
            # در غیر این صورت، بردار صفر باقی می‌ماند (padding)

        return {
            "trace_tensor": torch.tensor(trace_tensor_data, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "gate": gate  # نام گیت را به صورت رشته‌ای عبور می‌دهیم
        }


# -----------------
# (این بخش برای تست اسکریپت است)
if __name__ == "__main__":
    print("--- 🧪 شروع تست TrojanDataset ---")

    try:
        dataset = TrojanDataset(LABELED_DATA_FILE, EMBEDDING_FILE)

        print(f"\nتعداد کل نمونه‌ها: {len(dataset):,}")

        # یک نمونه را برای تست برمی‌داریم
        sample = dataset[0]

        print("\n--- نمونه اول دیتاست ---")
        print(f"Gate: {sample['gate']}")
        print(f"Label: {sample['label']}")
        print(f"Tensor Shape: {sample['trace_tensor'].shape}")
        print("✅ اسکریپت dataset.py به درستی کار می‌کند.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"❌ خطای ناشناخته در حین تست: {e}")