DeepNet-Trojan: A Deep Learning and NLP Pipeline for Hardware Trojan DetectionThis repository contains a complete Python implementation of the hardware security methodology presented in the IEEE paper: "Deep Learning-based Hardware Trojan Detection with Block-based Netlist Information Extraction."This project provides a full, end-to-end pipeline to train a deep learning model that can detect malicious circuits (Hardware Trojans - HTs) by analyzing their structure in gate-level netlist files.Based on the research paper:S. Yu, C. Gu, W. Liu and M. O'Neill, "Deep Learning-based Hardware Trojan Detection with Block-based Netlist Information Extraction," IEEE Transactions on Emerging Topics in Computing, 2021.DOI: 10.1109/TETC.2021.3116484 [cite: 1.pdf]🧠 Core Concept: Circuits as a LanguageThe fundamental idea of this project is to treat the structural layout of a hardware circuit as a human language. This allows us to apply powerful Natural Language Processing (NLP) and Deep Learning techniques.The pipeline works as follows:File (.v) -> Graph: A gate-level netlist (Verilog file) is parsed into a detailed pin-to-pin directed graph using networkx.Graph -> Sentences: We perform a Breadth-First Search (BFS) from every single gate (as a "center component") to extract all signal paths that pass through it. Each path becomes a "sentence" (known as a PCP Trace).Connections -> Words: Each connection within this path (e.g., InputPin -> Cell -> OutputPin) is treated as a "word" (a PCP Word).NLP Model (Net2Vec): A Word2Vec (Skip-gram) model is trained on millions of these "sentences" to build an embedding dictionary (net2vec.vectors). This dictionary maps each "word" (circuit connection) to a 100-dimension vector that represents its contextual meaning.DL Model (LSTM): An LSTM (Long Short-Term Memory) network is trained on these vectorized sentences. It learns to distinguish the "grammar" and "structure" of sentences from normal circuits versus those found in Hardware Trojans.Voter (Final Detection): To detect a Trojan in a new file, the model scans all traces. A Voter module aggregates the results. If a specific gate is the center of many "suspicious" traces, it is flagged as part of a Trojan.✨ FeaturesFull 4-Phase Pipeline: Implements all stages described in the paper:Netlist Parsing & Feature Extraction (Algorithm 1 & 2)Net2Vec (NLP) Word EmbeddingLSTM Detector TrainingComponent-Level Voter for Final EvaluationDynamic & Robust Parser: The parser (detector.py) is designed to handle multiple Verilog netlist formats, including:Explicit port mapping (e.g., .Q(wireA)) used in the TRIT-TC/TS training benchmarks.Implicit (positional) port mapping (e.g., (wireA, wireB)) used in standard ISCAS benchmarks.Cell Normalization: Includes a normalization map (NORM_MAP) to translate different cell library names (e.g., nnd2s1, nand2_1, nand) into a single generic token (e.g., NAND2). This makes the model more robust and "library-agnostic".Optimized for Large Datasets: The data loaders (dataset_upldated.py) are designed to handle massive datasets (7M+ samples) on low-RAM (16GB) machines by loading all data into memory and using num_workers=0.📂 Project Structure.
├── 📄 1.pdf                     # The source research paper
├── 📁 TRIT-TC/                  # Training Data (Combinational Trojans)
│   ├── 📁 c2670_T001/
│   │   ├── c2670_T001.v
│   │   └── log.txt
│   ├── 📁 original_designs/
│   │   └── ...
│   └── ...
├── 📁 TRIT-TS/                  # Training Data (Sequential Trojans)
│   ├── ...
│   └── 📁 original_designs/
│       └── ...
├── 📁 Test/                     # Folder for new test files (e.g., s713.v)
│
├── 🐍 netlist_parser.py         # (Phase 0) Base classes for parsing .v files.
├── 🐍 phase1_graph_utils.py     # (Phase 1) Graph conversion (Alg 1) & trace extraction (Alg 2).
├── 🐍 run_batch_extraction.py   # (Phase 1) Script to process all 'TRIT-TC'/'TRIT-TS' folders.
├── 🐍 process_originals.py      # (Phase 1) Script to process all 'original_designs' folders.
│
├── 🐍 preprocess_nlp.py         # (Phase 2) Gathers all traces into 'corpus_ALL.jsonl'
├── 🐍 train_nlp.py              # (Phase 2) Trains Net2Vec model -> 'net2vec.vectors'
├── 🐍 create_balanced_dataset.py# (Phase 2) Creates 'labeled_traces_BALANCED.jsonl'
│
├── 🐍 dataset_upldated.py       # (Phase 3) PyTorch Dataset class.
├── 🐍 model.py                  # (Phase 3) Defines the TrojanLSTM architecture.
├── 🐍 train_detector_updated.py # (Phase 3) Trains the LSTM model -> 'trojan_detector_final.pth'
│
├── 🐍 evaluate.py               # (Phase 4) Evaluates the trained model with the Voter.
├── 🐍 detector.py               # (FINAL TOOL) Standalone script to scan a single .v file.
│
├── requirements.txt          # Python dependencies
└── README.md                   # You are here
🚀 Quickstart & Usage1. InstallationClone this repository:git clone <your-repo-url>
cd DeepNet-Trojan
Create and activate a Python virtual environment:python -m venv .venv
.\.venv\Scripts\activate
Install all required libraries (including PyTorch with CUDA support):pip install -r requirements.txt
2. Option A: Re-Train The Entire Model (Recommended)This is the full pipeline to replicate the paper's results from scratch.# === Phase 1: Feature Extraction ===
# (This runs Alg 1 & 2 on all training files and creates .gpickle and .json files)
# (This will take a long time)
echo "--- Running Phase 1 ---"
python run_batch_extraction.py
python process_originals.py

# === Phase 2: NLP & Data Prep ===
# (This gathers all .json files and trains the language model)
echo "--- Running Phase 2 ---"
python preprocess_nlp.py
python train_nlp.py
python create_balanced_dataset.py

# === Phase 3: Train the Detector ===
# (This uses your GPU to train the LSTM)
echo "--- Running Phase 3 ---"
python train_detector_updated.py

# === Phase 4: Evaluate the Model ===
# (This runs the Voter and gives you the final accuracy metrics)
echo "--- Running Phase 4 ---"
python evaluate.py
3. Option B: Use the Pre-Trained Model (Detection Tool)Once you have the net2vec.vectors and trojan_detector_final.pth files, you can use detector.py as a standalone tool to scan new, unseen netlist files.Usage:python detector.py <path_to_your_netlist.v>
Example (scanning a known-good file):D:\VHDL_AI_Project> python detector.py D:\VHDL_AI_Project\Test\s713.v

--- 🔬 فاز 1: در حال پردازش s713.v ---
  (1/3) 📄 Parsing Netlist (Dynamic Mode)...
  (2/3) 🧱 Generating Blocks (Alg 1): 100%|...| 412/412
  (3/3) 💬 Extracting Traces (Alg 2): 100%|...| 412/412
✅ فاز 1 کامل شد. 1,095 ردیابی از 412 گیت استخراج شد.

--- 🧠 فاز 3: در حال بارگذاری مدل‌ها ---
  (استفاده از دستگاه: cuda)
  ✅ دیکشنری Net2Vec (vectors) بارگذاری شد.
  ✅ مدل آشکارساز (trojan_detector_final.pth) بارگذاری شد.

--- 🤖 فاز 4: در حال اجرای استنتاج و رأی‌گیری ---
  (1/2) 🧠 در حال استنتاج (Inference): 100%|...| 9/9
  (2/2) 🗳️ در حال رأی‌گیری (Voter): 100%|...| 412/412

==================================================
🏁 اسکن کامل شد
==================================================
  ✅ نتیجه: هیچ تروجان سخت‌افزاری در این فایل پیدا نشد.

⏱️ زمان کل اسکن: 12.52 ثانیه
Example (scanning a known-bad file):D:\VHDL_AI_Project> python detector.py D:\VHDL_AI_Project\Test\s38417_T0099_C.v

--- 🔬 فاز 1: در حال پردازش s38417_T0099_C.v ---
  ... (parsing and extraction) ...
✅ فاز 1 کامل شد. 10,000+ ردیابی استخراج شد.
...
--- 🤖 فاز 4: در حال اجرای استنتاج و رأی‌گیری ---
  ... (inference and voting) ...

==================================================
🏁 اسکن کامل شد
==================================================
  🚨 هشدار: 4 گیت مشکوک به تروجان پیدا شد!
--------------------------------------------------
  لیست گیت‌های مشکوک:
    1. troj49_0_U1
    2. troj49_0_U2
    3. troj49_0_U3
    4. troj49_0_U4

⏱️ زمان کل اسکن: 14.20 ثانیه
📚 CitationsPrimary Paper: S. Yu, et al. "Deep Learning-based Hardware Trojan Detection with Block-based Netlist Information Extraction." IEEE TETC, 2021.Benchmark Data: J. Cruz, et al. "An Automated Configurable Trojan Insertion Framework for Dynamic Trust Benchmarks" DATE, 2018. (Provided by Trust-Hub)📄 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.
