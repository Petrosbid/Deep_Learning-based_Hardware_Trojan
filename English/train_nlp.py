import os
import json
import gensim
import time
import logging

# --- Settings ---
CORPUS_FILE = "../jsonl_dataset/corpus_ALL.jsonl"

MODEL_OUTPUT_FILE = "../Model/net2vec.vectors"

# Model parameters, according to the paper
EMBEDDING_DIM = 100  # (N=100)
NEGATIVE_SAMPLES = 5  # (K=5)
WINDOW_SIZE = 9  # (2 * logic_level=5) - 1. A large window is considered
MIN_WORD_COUNT = 1  # Minimum repetition count for a word to be included
WORKERS = os.cpu_count() - 2  # Use all CPU cores except 2

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class CorpusStreamer:
    """
    A class for reading 48 million lines without loading into RAM.
    This class reads the .jsonl file line by line and yields each line.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Corpus file not found: {filepath}")

    def __iter__(self):
        print("\n--- 🧠 Starting streaming read of Corpus... (this process may take a long time) ---")
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
    Trains the Net2Vec (Word2Vec) model.
    """
    start_time = time.time()

    sentences_stream = CorpusStreamer(corpus_filepath)

    print(f"--- 🏋️ Starting Word2Vec Model Training ---")
    print(f"Parameters:")
    print(f"  Dimensions (vector_size): {EMBEDDING_DIM}")
    print(f"  Algorithm (sg): 1 (Skip-gram)")
    print(f"  Negative Sampling: {NEGATIVE_SAMPLES}")
    print(f"  Workers (CPU Cores): {WORKERS}")
    print("This process will take several hours and will use CPU...")

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
    print(f"\n--- ✅ Model Training Complete ---")
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes (or {(end_time - start_time) / 3600:.2f} hours)")

    # 3. Save model
    try:
        print(f"💾 Saving vector dictionary to {MODEL_OUTPUT_FILE}...")
        model.wv.save_word2vec_format(MODEL_OUTPUT_FILE)
        print("✅ ... saved.")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

    vocab_size = len(model.wv.index_to_key)
    print(f"\n--- 📊 Model Results ---")
    print(f"Total unique words (PCPs) in dictionary: {vocab_size}")
    if vocab_size > 0:
        print("First 10 words in dictionary:")
        for i, word in enumerate(model.wv.index_to_key[:10]):
            print(f"  {i + 1}. {word}")


def main():
    if not os.path.exists(CORPUS_FILE):
        print(f"❌ Error: {CORPUS_FILE} file not found.")
        print("Please run preprocess_nlp.py script first.")
        return

    train_net2vec_model(CORPUS_FILE)
    print("\n🏁 Phase 2 (NLP Training) Complete.")
    print(f"Next step: Creating balanced dataset (Downsampling) and training LSTM detector.")


if __name__ == "__main__":
    main()