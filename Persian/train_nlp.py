import os
import json
import gensim
import time
import logging

# --- تنظیمات ---
CORPUS_FILE = "../jsonl_dataset/corpus_ALL.jsonl"

MODEL_OUTPUT_FILE = "../Model/net2vec.vectors"

# پارامترهای مدل، مطابق با مقاله
EMBEDDING_DIM = 100  # (N=100)
NEGATIVE_SAMPLES = 5  # (K=5)
WINDOW_SIZE = 9  # (2 * logic_level=5) - 1. یک پنجره بزرگ در نظر می‌گیریم
MIN_WORD_COUNT = 1  # حداقل تعداد تکرار یک کلمه برای لحاظ شدن
WORKERS = os.cpu_count() - 2  # استفاده از تمام هسته‌های CPU بجز 2 تا

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class CorpusStreamer:
    """
    کلاسی برای خواندن 48 میلیون خط بدون بارگذاری در RAM.
    این کلاس فایل .jsonl را خط به خط می‌خواند و هر خط را yield می‌کند.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"فایل Corpus یافت نشد: {filepath}")

    def __iter__(self):
        print("\n--- 🧠 شروع خواندن جریانی Corpus... (این مرحله ممکن است طولانی باشد) ---")
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed line: {line}")
        except Exception as e:
            print(f"Error reading corpus file: {e}")
            raise


def train_net2vec_model(corpus_filepath):
    """
    مدل Net2Vec (Word2Vec) را آموزش می‌دهد.
    """
    start_time = time.time()

    sentences_stream = CorpusStreamer(corpus_filepath)

    print(f"--- 🏋️ شروع آموزش مدل Word2Vec ---")
    print(f"پارامترها:")
    print(f"  Dimensions (vector_size): {EMBEDDING_DIM}")
    print(f"  Algorithm (sg): 1 (Skip-gram)")
    print(f"  Negative Sampling: {NEGATIVE_SAMPLES}")
    print(f"  Workers (CPU Cores): {WORKERS}")
    print("این فرآیند چندین ساعت طول خواهد کشید و از CPU استفاده می‌کند...")

    model = gensim.models.Word2Vec(
        sentences=sentences_stream,
        vector_size=EMBEDDING_DIM,
        sg=1,
        negative=NEGATIVE_SAMPLES,
        window=WINDOW_SIZE,
        min_count=MIN_WORD_COUNT,
        workers=WORKERS,
        epochs=5
    )

    end_time = time.time()
    print(f"\n--- ✅ آموزش مدل کامل شد ---")
    print(f"زمان آموزش: {(end_time - start_time) / 60:.2f} دقیقه (یا {(end_time - start_time) / 3600:.2f} ساعت)")

    # 3. ذخیره مدل
    try:
        print(f"💾 در حال ذخیره دیکشنری بردارها در {MODEL_OUTPUT_FILE}...")
        model.wv.save_word2vec_format(MODEL_OUTPUT_FILE)
        print("✅ ... ذخیره شد.")
    except Exception as e:
        print(f"❌ خطا در ذخیره مدل: {e}")

    vocab_size = len(model.wv.index_to_key)
    print(f"\n--- 📊 نتایج مدل ---")
    print(f"تعداد کل کلمات (PCPs) یکتا در دیکشنری: {vocab_size}")
    if vocab_size > 0:
        print("10 کلمه اول در دیکشنری:")
        for i, word in enumerate(model.wv.index_to_key[:10]):
            print(f"  {i + 1}. {word}")


def main():
    if not os.path.exists(CORPUS_FILE):
        print(f"❌ خطا: فایل {CORPUS_FILE} یافت نشد.")
        print("لطفاً ابتدا اسکریپت preprocess_nlp.py را اجرا کنید.")
        return

    train_net2vec_model(CORPUS_FILE)
    print("\n🏁 فاز 2 (آموزش NLP) کامل شد.")
    print(f"مرحله بعدی: ساخت دیتاست متعادل (Downsampling) و آموزش آشکارساز LSTM.")


if __name__ == "__main__":
    main()