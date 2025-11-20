# **DeepNet-Trojan: A Deep Learning and NLP Pipeline for Hardware Trojan Detection**

This repository contains a complete Python implementation of the hardware security methodology presented in the IEEE paper: **"Deep Learning-based Hardware Trojan Detection with Block-based Netlist Information Extraction."**

This project provides a full, end-to-end pipeline to train a deep learning model that can detect malicious circuits (Hardware Trojans - HTs) by analyzing their structure in gate-level netlist files.

**Based on the research paper:**

S. Yu, C. Gu, W. Liu and M. O'Neill, "Deep Learning-based Hardware Trojan Detection with Block-based Netlist Information Extraction," IEEE Transactions on Emerging Topics in Computing, 2021.  
DOI: 10.1109/TETC.2021.3116484

## **🧠 Core Concept: Circuits as a Language**

The fundamental idea of this project is to treat the structural layout of a hardware circuit as a human language. This allows us to apply powerful Natural Language Processing (NLP) and Deep Learning techniques.

The pipeline works as follows:

1. **File (.v) -> Graph:** A gate-level netlist (Verilog file) is parsed into a detailed pin-to-pin directed graph using networkx.
2. **Graph -> Sentences:** We perform a Breadth-First Search (BFS) from every single gate (as a "center component") to extract all signal paths that pass through it. Each path becomes a "sentence" (known as a **PCP Trace**).
3. **Connections -> Words:** Each connection within this path (e.g., *InputPin -> Cell -> OutputPin*) is treated as a "word" (a **PCP Word**).
4. **NLP Model (Net2Vec):** A Word2Vec (Skip-gram) model is trained on millions of these "sentences" to build an embedding dictionary (net2vec.vectors). This dictionary maps each "word" (circuit connection) to a 100-dimension vector that represents its contextual meaning.
5. **DL Model (LSTM):** An LSTM (Long Short-Term Memory) network is trained on these vectorized sentences. It learns to distinguish the "grammar" and "structure" of sentences from normal circuits versus those found in Hardware Trojans.
6. **Voter (Final Detection):** To detect a Trojan in a new file, the model scans all traces. A **Voter** module aggregates the results. If a specific gate is the center of many "suspicious" traces, it is flagged as part of a Trojan.

## **✨ Features**

* **Full 4-Phase Pipeline:** Implements all stages described in the paper:
  1. Netlist Parsing & Feature Extraction (Algorithm 1 & 2)
  2. Net2Vec (NLP) Word Embedding
  3. LSTM Detector Training
  4. Component-Level Voter for Final Evaluation
* **Dynamic & Robust Parser:** The parser (detector.py) is designed to handle multiple Verilog netlist formats, including:
  * **Explicit** port mapping (e.g., .Q(wireA)) used in the TRIT-TC/TS training benchmarks.
  * **Implicit** (positional) port mapping (e.g., (wireA, wireB)) used in standard ISCAS benchmarks.
* **Cell Normalization:** Includes a normalization map (NORM_MAP) to translate different cell library names (e.g., nnd2s1, nand2_1, nand) into a single generic token (e.g., NAND2). This makes the model more robust and "library-agnostic".
* **Optimized for Large Datasets:** The data loaders (dataset.py) are designed to handle massive datasets (7M+ samples) on low-RAM (16GB) machines by loading all data into memory and using num_workers=0.

## **📂 Project Structure**

```
.
├── 📁 Dataset/                  # Contains training/testing data
├── 📁 English/                 # English implementation files and documentation
│   ├── 🐍 cleaner.py           # Data cleaning utilities
│   ├── 🐍 create_balanced_dataset.py  # Dataset balancing tools
│   ├── 🐍 dataset.py           # PyTorch dataset implementation
│   ├── 🐍 detector.py          # Main detection tool for scanning netlists
│   ├── 🐍 evaluate.py          # Model evaluation tools
│   ├── 🐍 model.py             # LSTM model definition
│   ├── 🐍 netlist_parser.py    # Netlist parsing utilities
│   ├── 🐍 phase1_graph_utils.py # Graph conversion and trace extraction
│   ├── 🐍 preprocess_nlp.py    # NLP preprocessing tools
│   ├── 🐍 process_originals.py # Processing original (non-trojan) designs
│   ├── 🐍 run_batch_extraction.py # Batch processing for feature extraction
│   ├── 🐍 train_detector_updated.py # LSTM model training
│   ├── 🐍 train_nlp.py         # Net2Vec model training
│   └── 📄 README.md            # This file
├── 📁 jsonl_dataset/           # JSONL formatted training data
│   ├── corpus_ALL.jsonl        # Complete corpus for NLP model
│   └── labeled_traces_BALANCED.jsonl  # Balanced training dataset
├── 📁 Model/                   # Trained models
│   ├── net2vec.vectors         # Trained word embeddings
│   └── trojan_detector_final.pth  # Trained LSTM detector
├── 📁 Persian/                 # Persian implementation files (duplicate of English)
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

## **🚀 Quickstart & Usage**

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

### **2. Option A: Re-Train The Entire Model (Recommended)**

This is the full pipeline to replicate the paper's results from scratch.

```bash
# === Phase 1: Feature Extraction ===
# (This runs Alg 1 & 2 on all training files and creates .gpickle and .json files)
# (This will take a long time)
echo "--- Running Phase 1 ---"
python English/run_batch_extraction.py
python English/process_originals.py

# === Phase 2: NLP & Data Prep ===
# (This gathers all .json files and trains the language model)
echo "--- Running Phase 2 ---"
python English/preprocess_nlp.py
python English/train_nlp.py
python English/create_balanced_dataset.py

# === Phase 3: Train the Detector ===
# (This uses your GPU to train the LSTM)
echo "--- Running Phase 3 ---"
python English/train_detector.py

# === Phase 4: Evaluate the Model ===
# (This runs the Voter and gives you the final accuracy metrics)
echo "--- Running Phase 4 ---"
python English/evaluate.py
```

### **3. Option B: Use the Pre-Trained Model (Detection Tool)**

Once you have the net2vec.vectors and trojan_detector_final.pth files, you can use detector.py as a standalone tool to scan new, unseen netlist files.

**Usage:**

```bash
python English/detector.py <path_to_your_netlist.v>
```

**Example (scanning a known-good file):**

```bash
python English/detector.py Test/s713.v
```

```
--- 🔬 Phase 1: Processing s713.v ---
  (1/3) 📄 Parsing Netlist (Dynamic Mode)...
  (2/3) 🧱 Generating Blocks (Alg 1): 100%|...| 412/412
  (3/3) 💬 Extracting Traces (Alg 2): 100%|...| 412/412
✅ Phase 1 completed. 1,095 traces extracted from 412 gates.

--- 🧠 Phase 3: Loading models ---
  (Using device: cuda)
  ✅ Net2Vec dictionary (vectors) loaded.
  ✅ Trojan Detector model (trojan_detector_final.pth) loaded.

--- 🤖 Phase 4: Running Inference and Voting ---
  (1/2) 🧠 Inference running: 100%|...| 9/9
  (2/2) 🗳️ Voting process: 100%|...| 412/412

==================================================
🏁 Scan complete
==================================================
  ✅ Result: No hardware trojans found in this file.

⏱️ Total scan time: 12.52 seconds
```

**Example (scanning a known-bad file):**

```bash
python English/detector.py Test/s38417_T0099_C.v
```

```
--- 🔬 Phase 1: Processing s38417_T0099_C.v ---
  ... (parsing and extraction) ...
✅ Phase 1 completed. 10,000+ traces extracted.
...
--- 🤖 Phase 4: Running Inference and Voting ---
  ... (inference and voting) ...

==================================================
🏁 Scan complete
==================================================
  🚨 Warning: 4 trojan-suspected gates found!
--------------------------------------------------
  List of suspicious gates:
    1. troj49_0_U1
    2. troj49_0_U2
    3. troj49_0_U3
    4. troj49_0_U4

⏱️ Total scan time: 14.20 seconds
```

## **📚 Citations**

* **Primary Paper:** S. Yu, et al. "Deep Learning-based Hardware Trojan Detection with Block-based Netlist Information Extraction." *IEEE TETC*, 2021.
* **Benchmark Data:** J. Cruz, et al. "An Automated Configurable Trojan Insertion Framework for Dynamic Trust Benchmarks" *DATE*, 2018. (Provided by [Trust-Hub](https://trust-hub.org/))

## **📄 License**

This project is licensed under the MIT License - see the LICENSE file for details.