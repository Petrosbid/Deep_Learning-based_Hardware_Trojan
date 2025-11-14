#
# evaluate.py
# (Final version with proper circuit-based splitting - no data leakage)
#
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import json
import random

# Import custom classes
try:
    from dataset import TrojanDataset, LABELED_DATA_FILE, EMBEDDING_FILE
    from model import TrojanLSTM
except ImportError:
    print("❌ Error: Make sure dataset.py and model.py files are in the same folder.")
    exit()

# --- Settings ---
BATCH_SIZE = 64
TRAIN_SPLIT = 0.8  # Must be the same as in train_detector.py
MODEL_FILE = "../Model/trojan_detector_final.pth"  # New model name
NUM_WORKERS = 4
random.seed(42)  # Use the same seed to ensure identical splitting


def get_unique_circuits(data_file):
    """
    (Copied from train_detector)
    Reads the .jsonl file once to extract the list of unique circuits.
    """
    print(f"--- 🔍 Reading {data_file} to find unique circuits...")
    circuits = set()
    with open(data_file, 'r', encoding='utf-8') as f:
        # For speed, we may not need to read the entire file,
        # but for complete assurance, we read the whole file.
        for line in tqdm(f, desc="🔍 Finding circuits"):
            try:
                if line.strip():
                    circuits.add(json.loads(line)['circuit'])
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"✅ {len(circuits)} unique circuits found.")
    return list(circuits)


def main():
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🚀 Using device: {device} ---")

    # --- ✨ 1. (New) Exact reconstruction of circuit-based splitting ---
    all_circuits = get_unique_circuits(LABELED_DATA_FILE)
    random.shuffle(all_circuits)  # seed=42 ensures the order is identical

    split_index = int(len(all_circuits) * TRAIN_SPLIT)
    # We only need the test (validation) circuits
    val_circuit_names = set(all_circuits[split_index:])

    print(f"\n--- 📊 Loading Validation Dataset (Test Set) ---")
    print(f"Number of test circuits: {len(val_circuit_names)}")

    # --- 2. (New) Build dataset only for test circuits ---
    print("\n(Loading validation dataset...)")
    val_dataset = TrojanDataset(LABELED_DATA_FILE, EMBEDDING_FILE,
                                allowed_circuits_list=val_circuit_names)

    print(f"Total test traces: {len(val_dataset):,}")

    # 3. Create DataLoader for test
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    # 4. Load trained model (new model)
    try:
        model = TrojanLSTM().to(device)
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()
        print(f"✅ Model successfully loaded from {MODEL_FILE}.")
    except FileNotFoundError:
        print(f"❌ Error: Model file {MODEL_FILE} not found.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("\n--- 🔬 Starting evaluation (steps 10 and 11 of paper) ---")

    # 5. Run model on test data
    gate_votes = defaultdict(lambda: {'true_label': 0, 'votes': []})

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="📊 Evaluating traces"):
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

            # 6. Collect votes for VOTER
            for i in range(len(gates)):
                gate_name = gates[i]
                gate_votes[gate_name]['votes'].append(preds_cpu[i])
                gate_votes[gate_name]['true_label'] = labels_cpu[i]

    print("✅ Trace evaluation completed.")
    print(f"🗳️ {len(gate_votes)} unique components for voting found.")

    # 7. Implement VOTER (step 12 of paper)
    print("\n--- 🗳️ Starting voting (Voter - step 12 of paper) ---")

    y_true_component = []  # True labels of gates
    y_pred_component = []  # Final predictions of gates

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

    print("✅ Voting completed.")

    # 8. Calculate final results (component level)
    print("\n" + "=" * 50)
    print("🏁 Final Results (Component Level - Correct Evaluation)")
    print("=" * 50)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true_component, y_pred_component).ravel()

        print(f"  True Positives (TP) - Found Trojans: {tp:,}")
        print(f"  False Negatives (FN) - Missed Trojans: {fn:,}")
        print(f"  True Negatives (TN) - Correctly Normal: {tn:,}")
        print(f"  False Positives (FP) - False Alarms: {fp:,}")
        print("-" * 50)

        # Calculate metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        print(f"  📊 Your Model's Final Metrics (No Data Leakage):")
        print(f"  - TPR (Recall / Accuracy): {tpr * 100:.2f}%")
        print(f"  - TNR (Specificity):     {tnr * 100:.2f}%")
        print(f"  - PPV (Precision):       {ppv * 100:.2f}%")
        print(f"  - NPV:                   {npv * 100:.2f}%")

        print("\n" + "-" * 50)
        print("  📖 Paper Results (for comparison):")
        print("  - Comb. (TC): 79.29% TPR, 99.97% TNR, 87.75% PPV, 99.94% NPV")
        print("  - Seq. (TS):  93.46% TPR, 99.99% TNR, 98.92% PPV, 99.92% NPV")

    except ValueError:
        print("❌ Error: It seems there were no samples in the validation set.")
    except Exception as e:
        print(f"❌ Error calculating results: {e}")

    total_time = time.time() - start_time
    print(f"\n⏱️ Total evaluation time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()