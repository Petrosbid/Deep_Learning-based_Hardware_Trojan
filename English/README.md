# **DeepNet-Trojan: A Deep Learning and Natural Language Processing Pipeline for Hardware Trojan Detection**

This repository contains a complete Python implementation of the hardware security approach presented in the IEEE paper: **"Deep Learning-Based Hardware Trojan Detection with Netlist Block-Based Information Extraction."**

This project provides a complete end-to-end pipeline to train a deep learning model that can identify malicious circuits (Hardware Trojan - HT) by analyzing their structure in gate-level netlist files.

**Based on research paper:**

S. Yu, C. Guo, V. Liu and M. O'Neil, "Deep Learning-Based Hardware Trojan Detection With Netlist Block-Based Information Extraction," IEEE Transactions on Emerging Topics in Computing, 2021.
DOI: 10.1109/TETC.2021.3116484

## **🧠 Core Concept: Circuits as a Language**

The fundamental idea of this project is to consider the layout structure of a hardware circuit as a human language. This allows us to leverage powerful Natural Language Processing (NLP) and Deep Learning techniques.

The pipeline works as follows:

1. **(.v file) -> Graph:** A gate-level netlist (Verilog file) is parsed into a directed pin-level graph using networkx.
2. **Graph -> Sentences:** We perform a breadth-first search (BFS) from each single gate (as a "central component") to extract all signal paths that pass through it. Each path becomes a "sentence" (known as a **PCP trace**).
3. **Connections -> Words:** Each connection in this path (e.g., *InputPin -> Cell -> OutputPin*) is treated as a "word" (a **PCP word**).
4. **NLP Model (Net2Vec):** A Word2Vec (Skip-gram) model is trained on millions of such "sentences" to create a vocabulary (net2vec.vectors). This vocabulary maps each "word" (circuit connection) to a 100-dimensional vector that represents its semantic meaning.
5. **DL Model (LSTM):** An LSTM (Long Short-Term Memory) network is trained on these vectorized sentences. It learns to distinguish the "grammar" and "structure" of normal circuit sentences from those found in Hardware Trojans.
6. **Voting (Final Detection):** For Trojan detection on a new file, the model scans all traces. A **voting module** aggregates the results. If a specific gate is the center of many "suspicious" traces, it is marked as part of a Trojan.

## **✨ Features**

* **Complete 4-Phase Pipeline:** Implements all stages described in the paper:
  1. Netlist feature decomposition and extraction (Algorithms 1 and 2)
  2. NLP word embedding (Net2Vec)
  3. LSTM detector training
  4. Component-level voting for final evaluation
* **Dynamic and Robust Parser:** The parser (detector.py) is designed to handle multiple Verilog netlist formats, including:
  * **Explicit gate mapping** (e.g. .Q(wireA)) used in TRIT-TC/TS training sets.
  * **Implicit gate mapping** (e.g. (wireA, wireB)) used in standard ISCAS sets.
* **Gate Normalization:** Includes a normalization map (NORM_MAP) to translate various library gate names (e.g. nnd2s1, nand2_1, nand) to a common label (e.g. NAND2). This makes the model more robust and "library-agnostic".
* **Optimized for Large Datasets:** Data loaders (dataset.py) are designed to handle large datasets (7 million+ samples) on machines with limited RAM (16GB) by loading all data into memory and using num_workers=0.

## **📂 Project Structure**

```
.
├── 📁 Dataset/                  # Contains training/testing data
├── 📁 English/                 # English implementation files and documentation
│   ├── 🐍 cleaner.py           # Data cleaning tools
│   ├── 🐍 create_balanced_dataset.py  # Data balancing tools
│   ├── 🐍 dataset.py           # PyTorch Dataset implementation
│   ├── 🐍 detector.py          # Main detection tool for scanning netlists
│   ├── 🐍 evaluate.py          # Model evaluation tools
│   ├── 🐍 model.py             # LSTM model definition
│   ├── 🐍 netlist_parser.py    # Netlist parsing tools
│   ├── 🐍 phase1_graph_utils.py # Graph conversion and trace extraction
│   ├── 🐍 preprocess_nlp.py    # NLP preprocessing tools
│   ├── 🐍 process_originals.py # Processing original designs (without trojans)
│   ├── 🐍 run_batch_extraction.py # Batch processing for feature extraction
│   ├── 🐍 train_detector_updated.py # LSTM model training
│   ├── 🐍 train_nlp.py         # Net2Vec model training
│   └── 📄 README.md            # This file
├── 📁 jsonl_dataset/           # Training data in JSONL format
│   ├── corpus_ALL.jsonl        # Complete corpus for NLP model
│   └── labeled_traces_BALANCED.jsonl  # Balanced training dataset
├── 📁 Model/                   # Trained models
│   ├── net2vec.vectors         # Trained word embeddings
│   └── trojan_detector_final.pth  # Trained LSTM detector
├── 📁 Persian/                 # Persian implementation files (redundant from English)
│   ├── 🐍 cleaner.py
│   ├── 🐍 create_balanced_dataset.py
│   ├── 🐍 dataset.py
│   ├── 🐍 detector.py
│   ├── 🐍 evaluate.py
│   ├── 🐍 model.py
│   ├── 🐍 netlist_parser.py
│   ├── 🐍 phase1_graph_utils.py
│   ├── 🐍 preprocess_nlp.py
│   ├── 🐍 process_originals.py
│   ├── 🐍 run_batch_extraction.py
│   ├── 🐍 train_detector_updated.py
│   └── 🐍 train_nlp.py
└── 📄 .gitignore               # Git ignore patterns
```

## **🚀 Quick Start and Usage**

### **1. Installation**

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd VHDL_AI_Project
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```

3. Install all required libraries:
   ```bash
   pip install torch torchvision torchaudio
   pip install networkx gensim tqdm numpy
   ```

### **2. Option A: Retrain the entire model (recommended)**

This complete pipeline is for reproducing the paper's results from scratch.

```bash
# === Phase 1: Feature Extraction ===
# (This runs Algorithms 1 and 2 on all training files and creates .gpickle and .json files)
# (This will take a long time)
echo "--- Running Phase 1 ---"
python English/run_batch_extraction.py
python English/process_originals.py

# === Phase 2: Natural Language Processing and Data Preparation ===
# (This collects all .json files and trains the language model)
echo "--- Running Phase 2 ---"
python English/preprocess_nlp.py
python English/train_nlp.py
python English/create_balanced_dataset.py

# === Phase 3: Detector Training ===
# (This uses your GPU to train the LSTM)
echo "--- Running Phase 3 ---"
python English/train_detector.py

# === Phase 4: Model Evaluation ===
# (This runs voting and provides final accuracy metrics for you)
echo "--- Running Phase 4 ---"
python English/evaluate.py
```

### **3. Option B: Use the trained model (detection tool)**

After having net2vec.vectors and trojan_detector_final.pth, you can use detector.py as a standalone tool to scan new netlist files.

**Usage:**

```bash
python English/detector.py <path_to_your_netlist.v>
```

**Example (scanning a known clean file):**

```bash
python English/detector.py Test/s713.v
```

```
--- 🔬 Phase 1: Processing s713.v ---
  (1/3) 📄 Parsing Netlist (Dynamic Mode)...
  (2/3) 🧱 Generating Blocks (Algorithm 1): 100%|...| 412/412
  (3/3) 💬 Extracting Traces (Algorithm 2): 100%|...| 412/412
✅ Phase 1 Complete. 1,095 traces extracted from 412 gates.

--- 🧠 Phase 3: Loading Models ---
  (Using device: cuda)
  ✅ Net2Vec Vocabulary (vectors) loaded.
  ✅ Trojan Detection Model (trojan_detector_final.pth) loaded.

--- 🤖 Phase 4: Running Inference and Voting ---
  (1/2) 🧠 Running Inference: 100%|...| 9/9
  (2/2) 🗳️ Running Voting Process: 100%|...| 412/412

==================================================
🏁 Scan Complete
==================================================
  ✅ Result: No Hardware Trojan found in this file.

⏱️ Total scan time: 12.52 seconds
```

**Example (scanning a known bad file):**

```bash
python English/detector.py Test/s38417_T0099_C.v
```

```
--- 🔬 Phase 1: Processing s38417_T0099_C.v ---
  ... (parsing and extraction) ...
✅ Phase 1 Complete. Over 10,000 traces extracted.
...
--- 🤖 Phase 4: Running Inference and Voting ---
  ... (inference and voting) ...

==================================================
🏁 Scan Complete
==================================================
  🚨 Warning: 4 suspicious trojan gates found!
--------------------------------------------------
  List of Suspicious Gates:
    1. troj49_0_U1
    2. troj49_0_U2
    3. troj49_0_U3
    4. troj49_0_U4

⏱️ Total scan time: 14.20 seconds
```

## **📚 References**

* **Original paper:** S. Yu, et al. "Deep Learning-Based Hardware Trojan Detection with Netlist Block-Based Information Extraction." *IEEE TETC*, 2021.
* **Reference data:** J. Cruz, et al. "A Configurable Automatic Framework for Dynamic Trojan Insertion in Trust-Hub Benchmark Datasets" *DATE*, 2018. (Provided by [Trust-Hub](https://trust-hub.org/))

## **📄 License**

This project is licensed under the MIT License - see the license file for details.