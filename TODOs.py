
##################################
##############Phase 1#############
##################################
# TODO 1: Implement Cell-Pin Splitter (in graph_utils.py)
    # [ ] def convert_to_pin_graph(netlist: Netlist) -> PinGraph:
    # [ ]   Create PinGraph class (nodes: CellVertex, PinVertex; edges: directed)
    # [ ]   Logic:
    # [ ]     1. Create wire_map dict: {wire_name: [list_of_pins]}
    # [ ]     2. Iterate netlist.gates:
    # [ ]       - Create CellVertex (e.g., 'U1')
    # [ ]       - For each connection: Create PinVertex (e.g., 'U1_Q')
    # [ ]       - Add edge CellVertex <-> PinVertex (check direction)
    # [ ]       - Add PinVertex to wire_map
    # [ ]     3. Iterate wire_map:
    # [ ]       - Add directed edges between PinVertices sharing the same wire
    # [ ]   Output: PinGraph object (Adjacency List/Dict)

# TODO 2: BFS-based Netlist Block Generator (in graph_utils.py)
    # [ ] Implement Algorithm 1 Article
    # [ ] def generate_netlist_blocks(pin_graph: PinGraph, logic_level: int = 4) -> dict:
    # [ ]   Logic:
    # [ ]     1. Create def RECURSION(node, cnt, Direction)
    # [ ]     2. For each CellVertex in pin_graph:
    # [ ]       - block['I'] = RECURSION(cell, logic_level, 'I')
    # [ ]       - block['O'] = RECURSION(cell, logic_level, 'O')
    # [ ]       - Store in main dict
    # [ ]   Output: {cell_name: netlist_block_tree}

# TODO 3: Implement Feature Traces Extractor (in graph_utils.py)
    # [ ] Implement Algorithm 2 Article
    # [ ] def extract_pcp_traces(netlist_blocks: dict) -> dict:
    # [ ]   Logic:
    # [ ]     1. Implement def FLATTEN(block_tree) -> list_of_paths
    # [ ]     2. For each path in list_of_paths:
    # [ ]       - Convert path to PCP "words"
    # [ ]       - A "word" string = "vin_p" + "_" + "v_cell" + "_" + "vout_p"
    # [ ]       - Store list of "words" as a "trace" (sentence)
    # [ ]   Output eg: {'U1': [['DIN1_nnd4_Q', 'DIN_dff_Q'], ['DIN2_nnd4_Q', 'DIN_dff_Q']]}

##################################
##############Phase 2#############
##################################
# TODO 4: write preprocess.py script
    # [ ] import os, json, glob
    # [ ] import netlist_parser, graph_utils
    # [ ] Define DATASET_PATH
    # [ ] def find_all_files() -> list[tuple(str, str)]:
    # [ ]   Logic: Find all (.v, .log) file pairs in DATASET_PATH
    # [ ] def main():
    # [ ]   corpus = []
    # [ ]   labeled_trace_data = [] # For use in Phase 3
    # [ ]   for (netlist_file, log_file) in find_all_files():
    # [cite_start][ ]     trojan_names = netlist_parser.parse_trojan_log(log_file) [cite: 361-365]
    # [cite_start][ ]     netlist = netlist_parser.parse_netlist(netlist_file, trojan_names) [cite: 1-360]
    # [ ]     pin_graph = graph_utils.convert_to_pin_graph(netlist)
    # [ ]     net_blocks = graph_utils.generate_netlist_blocks(pin_graph, logic_level=4)
    # [ ]     all_traces_dict = graph_utils.extract_pcp_traces(net_blocks)
    # [ ]     # --- Store data for both NLP and DL training ---
    # [ ]     for center_gate, traces in all_traces_dict.items():
    # [ ]       label = 1 if center_gate in trojan_names else 0
    # [ ]       for trace in traces:
    # [ ]         corpus.append(trace) # For Net2Vec training
    # [ ]         labeled_trace_data.append({'trace': trace, 'label': label, 'gate': center_gate})
    # [ ]   # --- Save outputs ---
    # [ ]   with open('corpus.json', 'w') as f: json.dump(corpus, f)
    # [ ]   with open('labeled_traces.json', 'w') as f: json.dump(labeled_trace_data, f)
    # [ ] Output: corpus.json (List[List[str]])
    # [ ] Output: labeled_traces.json (List[Dict])

# TODO 5: write train_nlp.py script
    # [ ] import json, gensim
    # [ ] def load_corpus(file_path='corpus.json') -> list:
    # [ ]   with open(file_path, 'r') as f: return json.load(f)
    # [ ] def train_word2vec(corpus):
    # [ ]   model = gensim.models.Word2Vec(
    # [ ]       sentences=corpus,
    # [ ]       vector_size=100,  # N=100
    # [ ]       sg=1,             # Use Skip-gram
    # [ ]       negative=5,       # K=5
    # [ ]       window=9,         # Max trace length (e.g., 2*ll-1 for ll=5)
    # [ ]       min_count=1,
    # [ ]       epochs=5          # Train for 5 epochs
    # [ ]   )
    # [ ]   model.wv.save_word2vec_format('net2vec.vectors')
    # [ ] main():
    # [ ]   corpus = load_corpus()
    # [ ]   train_word2vec(corpus)
    # [ ] Output: net2vec.vectors (The trained embedding dictionary)

##################################
##############Phase 3#############
##################################
# TODO 6: Implement PyTorch Dataset (in dataset.py)
    # [ ] import torch, json, numpy as np
    # [ ] from torch.utils.data import Dataset
    # [ ] from gensim.models import KeyedVectors
    # [ ] class NetlistDataset(Dataset):
    # [ ]   def __init__(self, labeled_traces_path, embedding_path, max_len=7): # max_len = 2*ll-1
    # [ ]     self.data = json.load(open(labeled_traces_path, 'r'))
    # [ ]     self.embeddings = KeyedVectors.load_word2vec_format(embedding_path)
    # [ ]     self.max_len = max_len
    # [ ]   def __len__(self):
    # [ ]     return len(self.data)
    # [ ]   def __getitem__(self, idx):
    # [ ]     item = self.data[idx]
    # [ ]     trace, label, gate = item['trace'], item['label'], item['gate']
    # [ ]     # --- Vectorize and Pad Trace ---
    # [ ]     vectorized_trace = np.zeros((self.max_len, 100)) # 100 = embedding_size
    # [ ]     for i, word in enumerate(trace):
    # [ ]       if i >= self.max_len: break
    # [ ]       if word in self.embeddings:
    # [ ]         vectorized_trace[i] = self.embeddings[word]
    # [ ]     return {
    # [ ]         'trace': torch.tensor(vectorized_trace, dtype=torch.float32),
    # [ ]         'label': torch.tensor(label, dtype=torch.long),
    # [ ]         'gate': gate
    # [ ]     }

# TODO 7: Build LSTM Model (in model.py)
    # [ ] import torch.nn as nn
    # [ ] class TrojanLSTM(nn.Module):
    # [ ]   def __init__(self, input_size=100, hidden_size=128, num_layers=2, output_size=2):
    # [ ]     super(TrojanLSTM, self).__init__()
    # [ ]     # 2 LSTM layers, 128 hidden dim
    # [ ]     self.lstm = nn.LSTM(
    # [ ]         input_size=input_size,
    # [ ]         hidden_size=hidden_size,
    # [ ]         num_layers=num_layers,
    # [ ]         batch_first=True # Input shape is (batch, seq_len, features)
    # [ ]     )
    # [ ]     self.fc = nn.Linear(hidden_size, output_size) # FC layer
    # [ ]   def forward(self, x):
    # [ ]     # x shape: (batch_size, max_len, 100)
    # [ ]     lstm_out, (h_n, c_n) = self.lstm(x)
    # [ ]     # We only need the output of the last time step from the last layer
    # [ ]     last_hidden_state = h_n[-1] # shape: (batch_size, 128)
    # [ ]     out = self.fc(last_hidden_state) # shape: (batch_size, 2)
    # [ ]     return out

# TODO 8: Write Training Script (train_detector.py)
    # [ ] import torch
    # [ ] from torch.utils.data import DataLoader, random_split
    # [ ] from torch.utils.data.sampler import WeightedRandomSampler
    # [ ] from dataset import NetlistDataset
    # [ ] from model import TrojanLSTM
    # [ ]
    # [ ] # --- 1. Load Data ---
    # [ ] dataset = NetlistDataset('labeled_traces.json', 'net2vec.vectors', max_len=7)
    # [ ] train_size = int(0.8 * len(dataset))
    # [ ] test_size = len(dataset) - train_size
    # [ ] train_set, test_set = random_split(dataset, [train_size, test_size])
    # [ ]
    # [ ] # --- 2. Implement Upsampling for train_set ---
    # [ ] labels = [dataset.data[i]['label'] for i in train_set.indices]
    # [ ] class_counts = np.bincount(labels)
    # [ ] weights = 1. / class_counts
    # [ ] sample_weights = weights[labels]
    # [ ] sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    # [ ]
    # [ ] # --- 3. Create DataLoaders ---
    # [ ] train_loader = DataLoader(train_set, batch_size=32, sampler=sampler)
    # [ ] test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    # [ ]
    # [ ] # --- 4. Initialize Model, Loss, Optimizer ---
    # [ ] device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # [ ] model = TrojanLSTM().to(device)
    # [ ] criterion = nn.CrossEntropyLoss() # Use CrossEntropy
    # [ ] optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # [ ]
    # [ ] # --- 5. Training Loop ---
    # [ ] for epoch in range(5): # Train for 5 epochs
    # [ ]   model.train()
    # [ ]   for batch in train_loader:
    # [ ]     traces = batch['trace'].to(device)
    # [ ]     labels = batch['label'].to(device)
    # [ ]     optimizer.zero_grad()
    # [ ]     outputs = model(traces)
    # [ ]     loss = criterion(outputs, labels)
    # [ ]     loss.backward()
    # [ ]     optimizer.step()
    # [ ]   print(f"Epoch {epoch+1}/5, Loss: {loss.item()}")
    # [ ]
    # [ ] # --- 6. Save Model ---
    # [ ] torch.save(model.state_dict(), 'trojan_detector_lstm.pth')
    # [ ] torch.save(test_loader, 'test_loader.pth') # Save test_loader for evaluation
    # [ ] Output: trojan_detector_lstm.pth
    # [ ] Output: test_loader.pth

##################################
##############Phase 4#############
##################################
# TODO 9: Write Evaluation Script (evaluate.py)
    # [ ] import torch, numpy as np
    # [ ] from sklearn.metrics import confusion_matrix
    # [ ] from model import TrojanLSTM
    # [ ]
    # [ ] # --- 1. Load Model and Test Data ---
    # [ ] device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # [ ] model = TrojanLSTM().to(device)
    # [ ] model.load_state_dict(torch.load('trojan_detector_lstm.pth'))
    # [ ] model.eval() # Set model to evaluation mode
    # [ ] test_loader = torch.load('test_loader.pth')
    # [ ]
    # [ ] # --- 2. Get Trace-Level Predictions (Gams 10, 11) ---
    # [ ] trace_predictions = [] # Store (gate, pred, true_label)
    # [ ] with torch.no_grad():
    # [ ]   for batch in test_loader:
    # [ ]     traces = batch['trace'].to(device)
    # [ ]     labels = batch['label'].numpy()
    # [ ]     gates = batch['gate']
    # [ ]
    # [ ]     outputs = model(traces)
    # [ ]     _, predicted = torch.max(outputs.data, 1)
    # [ ]     predicted = predicted.cpu().numpy()
    # [ ]
    # [ ]     for i in range(len(gates)):
    # [ ]       trace_predictions.append((gates[i], predicted[i], labels[i]))
    # [ ]
    # [ ] # --- 3. Implement Voter (Gam 12) ---
    # [ ] votes = {} # {gate_name: {'HT': 0, 'Normal': 0, 'TrueLabel': 0}}
    # [ ] for gate, pred, true_label in trace_predictions:
    # [ ]   if gate not in votes:
    # [ ]     votes[gate] = {'HT': 0, 'Normal': 0, 'TrueLabel': true_label}
    # [ ]   if pred == 1:
    # [ ]     votes[gate]['HT'] += 1
    # [ ]   else:
    # [ ]     votes[gate]['Normal'] += 1
    # [ ]
    # [ ] # --- 4. Get Final Component-Level Decisions ---
    # [ ] y_true_component = []
    # [ ] y_pred_component = []
    # [ ] for gate, counts in votes.items():
    # [ ]   y_true_component.append(counts['TrueLabel'])
    # [ ]   if counts['HT'] > counts['Normal']:
    # [ ]     y_pred_component.append(1)
    # [ ]   elif counts['Normal'] > counts['HT']:
    # [ ]     y_pred_component.append(0)
    # [ ]   else:
    # [ ]     y_pred_component.append(1) # Boundary condition
    # [ ]
    # [ ] # --- 5. Calculate Final Metrics ---
    # [ ] tn, fp, fn, tp = confusion_matrix(y_true_component, y_pred_component).ravel()
    # [ ]
    # [ ] TPR = tp / (tp + fn) #
    # [ ] TNR = tn / (tn + fp) #
    # [ ] PPV = tp / (tp + fp) #
    # [ ] NPV = tn / (tn + fn) #
    # [ ]
    # [ ] print("--- Component-Level Results (After Voting) ---")
    # [ ] print(f"TPR (Accuracy): {TPR * 100:.2f}%")
    # [ ] print(f"TNR: {TNR * 100:.2f}%")
    # [ ] print(f"PPV (Precision): {PPV * 100:.2f}%")
    # [ ] print(f"NPV: {NPV * 100:.2f}%")
    # [ ] print("-------------------------------------------------")
    # [ ] print(f"Compare with paper (Comb.): 79.29% TPR, 87.75% PPV")
    # [ ] print(f"Compare with paper (Seq.): 93.46% TPR, 98.92% PPV")

# TODO 10: (Optional) Implement CNN Model
    # [ ] Create model_cnn.py
    # [ ] class TrojanCNN(nn.Module):
    # [ ]   # Implement architecture from Fig. 11 and Table 2
    # [ ]   # (e.g., Conv2d -> MaxPool2d -> Conv2d -> MaxPool2d -> FC -> FC -> FC)
    # [ ]
    # [ ] (Modify train_detector.py and evaluate.py to use this model)