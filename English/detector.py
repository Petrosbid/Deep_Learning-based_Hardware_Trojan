#
# detector.py
# (Final Trojan Detection Tool - Enhanced with Dynamic Parser)
#
import os
import re
import json
import time
import pickle
import argparse
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any

# --- Settings ---
DEFAULT_MODEL_FILE = "../Model/trojan_detector_final.pth"
DEFAULT_VECTORS_FILE = "../Model/net2vec.vectors"
LOGIC_LEVEL = 4
MAX_TRACE_LENGTH = (2 * LOGIC_LEVEL) - 1
EMBEDDING_DIM = 100
BATCH_SIZE = 128
NUM_WORKERS = 0

# ##################################################################
# ########## ✨ (Enhanced) Cell Library ✨ ###################
# ##################################################################
# This library now only includes *explicit* gates (from training data).
# Implicit gates (like and, or, not) will be added dynamically.
# ##################################################################
# ########## ✨ (Final) Standard Cell Library ✨ ################
# ##################################################################
# This library now includes TRIT, ISCAS gate definitions and the new library (SAED)
CELL_LIBRARY = {
    # --- (New) SAED Library Gates (like s38417_T0099_C.v) ---
    # Simple gates
    'inv_1': (['O'], ['I']),
    'nand2_1': (['O'], ['I0', 'I1']),
    'nand3_1': (['O'], ['I0', 'I1', 'I2']),
    'nand4_1': (['O'], ['I0', 'I1', 'I2', 'I3']),
    'nor2_1': (['O'], ['I0', 'I1']),
    'nor3_1': (['O'], ['I0', 'I1', 'I2']),
    'nor4_1': (['O'], ['I0', 'I1', 'I2', 'I3']),
    'and2_0': (['O'], ['I0', 'I1']),
    'and3_1': (['O'], ['I0', 'I1', 'I2']),
    'xor2_1': (['O'], ['I0', 'I1']),

    # Complex gates (AOI/OAI/MUX)
    'mux2i_1': (['O'], ['I0', 'I1', 'S']),  # 2-input MUX
    'a21oi_1': (['O'], ['A1', 'A2', 'B1']),  # (A1&A2) | B1 -> Inverted
    'a22oi_1': (['O'], ['A1', 'A2', 'B1', 'B2']),  # (A1&A2) | (B1&B2) -> Inverted
    'a221oi_1': (['O'], ['A1', 'A2', 'B1', 'B2', 'C1']),  # (A1&A2) | (B1&B2) | C1 -> Inverted
    'a222oi_1': (['O'], ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']),  # ...
    'o21ai_0': (['O'], ['A1', 'A2', 'B1']),  # (A1|A2) & B1 -> Inverted
    'o32ai_1': (['O'], ['A1', 'A2', 'A3', 'B1', 'B2']),  # (A1|A2|A3) & (B1|B2) -> Inverted
    'o22ai_1': (['O'], ['A1', 'A2', 'B1', 'B2']),  # (A1|A2) & (B1|B2) -> Inverted
    'o221ai_1': (['O'], ['A1', 'A2', 'B1', 'B2', 'C1']),  # (A1|A2) & (B1|B2) & C1 -> Inverted
    # --------------------------------------------------

    # --- (Previous) General ISCAS Cells ---
    'dff': (['Q'], ['D', 'CLK']),
    'not': (['O'], ['I']),
    'and': (['O'], ['I0', 'I1']),
    'or': (['O'], ['I0', 'I1']),
    'nand': (['O'], ['I0', 'I1']),
    'nor': (['O'], ['I0', 'I1']),
    'and3': (['O'], ['I0', 'I1', 'I2']),
    'and4': (['O'], ['I0', 'I1', 'I2', 'I3']),
    'or3': (['O'], ['I0', 'I1', 'I2']),
    'or4': (['O'], ['I0', 'I1', 'I2', 'I3']),
    'nand3': (['O'], ['I0', 'I1', 'I2']),
    'nand4': (['O'], ['I0', 'I1', 'I2', 'I3']),
    'nor3': (['O'], ['I0', 'I1', 'I2']),
    'nor4': (['O'], ['I0', 'I1', 'I2', 'I3']),
    'dff_': (['Q', 'QN'], ['D', 'CLK', 'PRE', 'CLR']),
    'nand2': (['O'], ['I0', 'I1']),
    'inv': (['O'], ['I']),
    'and2': (['O'], ['I0', 'I1']),
    'nor2': (['O'], ['I0', 'I1']),
    'xor2': (['O'], ['I0', 'I1']),
    'buf': (['O'], ['I']),

    # --- TRIT Library Cells (Training Data) ---
    'nor2s1': (['Q'], ['DIN1', 'DIN2']),
    'hi1s1': (['Q'], ['DIN']),
    'nnd2s1': (['Q'], ['DIN1', 'DIN2']),
    'nor3s1': (['Q'], ['DIN1', 'DIN2', 'DIN3']),
    'nnd4s1': (['Q'], ['DIN1', 'DIN2', 'DIN3', 'DIN4']),
    'i1s1': (['Q'], ['DIN']),
    'or5s1': (['Q'], ['DIN1', 'DIN2', 'DIN3', 'DIN4', 'DIN5']),
    'xor2s1': (['Q'], ['DIN1', 'DIN2']),
    'dffs1': (['Q', 'QN'], ['DIN', 'CLK', 'EB']),
    'mux2s1': (['Q'], ['I0', 'I1', 'S']),
    'dffle': (['Q', 'QN'], ['DIN', 'CLK', 'EB']),
    'assign': (['Q'], ['DIN']),
}


# ##################################################################
# ########## Copied from model.py (unchanged) #####################
# ##################################################################
class TrojanLSTM(nn.Module):
    def __init__(self, input_size=100, hidden_size=128, num_layers=2, output_size=2):
        super(TrojanLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        last_hidden_state = h_n[-1]
        out = self.fc(last_hidden_state)
        return out


# ##################################################################
# ########## Copied from netlist_parser.py (Enhanced) #############
# ##################################################################
class Gate:
    def __init__(self, instance_name, cell_type):
        self.instance_name = instance_name
        self.cell_type = cell_type
        self.connections = {}

    def add_connection(self, port, wire): self.connections[port] = wire


class Netlist:
    def __init__(self, module_name):
        self.module_name = module_name;
        self.inputs = set();
        self.outputs = set()
        self.wires = set();
        self.gates = {}

    def add_gate(self, gate_obj): self.gates[gate_obj.instance_name] = gate_obj


def parse_netlist(netlist_file_path: str) -> Optional[Netlist]:
    trojan_gate_names = set()
    re_module = re.compile(r'^\s*module\s+([\w\d_]+)', re.IGNORECASE)
    re_port = re.compile(r'^\s*(input|output)\s+(.+);', re.IGNORECASE)
    re_wire = re.compile(r'^\s*wire\s+(.+);', re.IGNORECASE)
    re_assign = re.compile(r'^\s*assign\s+([\w\d_\[\]:]+)\s*=\s*([\w\d_\[\]:\'b]+);', re.IGNORECASE)
    re_gate = re.compile(r'^\s*([\w\d_]+)\s+([\w\d_:]+)\s*\((.*?)\);', re.DOTALL)
    re_port_conn = re.compile(r'\.([\w\d_]+)\s*\(([\w\d_\[\]:\'b]+)\)')
    netlist_obj = None;
    buffer = "";
    assign_counter = 0
    try:
        with open(netlist_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'): continue
                buffer += " " + line
                if not buffer.endswith(';'): continue
                complete_line = buffer.strip();
                buffer = ""
                if netlist_obj is None:
                    match = re_module.search(complete_line)
                    if match: netlist_obj = Netlist(module_name=match.group(1))
                    continue
                match = re_port.search(complete_line)
                if match:
                    port_type, ports_str = match.groups()
                    ports = [p.strip() for p in ports_str.split(',') if p.strip()]
                    if port_type.lower() == 'input':
                        netlist_obj.inputs.update(ports)
                    else:
                        netlist_obj.outputs.update(ports)
                    continue
                match = re_wire.search(complete_line)
                if match:
                    wires_str = match.group(1)
                    wires = [w.strip() for w in wires_str.split(',') if w.strip()]
                    netlist_obj.wires.update(wires)
                    continue
                match = re_assign.search(complete_line)
                if match:
                    dest_wire, source_wire = match.groups()
                    instance_name = f"assign_{assign_counter}";
                    assign_counter += 1
                    gate_obj = Gate(instance_name, 'assign')
                    gate_obj.add_connection('Q', dest_wire);
                    gate_obj.add_connection('DIN', source_wire)
                    netlist_obj.add_gate(gate_obj)
                    continue

                # --- ✨✨✨ Start Dynamic Parser Logic ✨✨✨ ---
                match = re_gate.search(complete_line)
                if match:
                    cell_type, instance_name, connections_str = match.groups()

                    instance_name = instance_name.replace(':', '').strip()
                    if instance_name.upper().startswith('G'):
                        instance_name = instance_name[1:]

                    gate_obj = Gate(instance_name, cell_type)

                    # Check mapping type: Explicit (.port) or Implicit (wire1, wire2)
                    if connections_str.strip().startswith('.'):
                        # --- 1. Explicit mode (like training data) ---
                        for port_match in re_port_conn.finditer(connections_str):
                            gate_obj.add_connection(port_match.group(1), port_match.group(2))
                    else:
                        # --- 2. Implicit mode (like s713.v) ---
                        wires = [w.strip() for w in connections_str.split(',') if w.strip()]

                        if not wires: continue  # Gate without connections

                        # --- ✨ Dynamic Logic: Assume first is output, rest are inputs ---
                        output_pin_name = "O"
                        gate_obj.add_connection(output_pin_name, wires[0])

                        input_pins = []
                        for i, input_wire in enumerate(wires[1:]):
                            input_pin_name = f"I{i}"  # Dynamic naming: I0, I1, I2, ...
                            gate_obj.add_connection(input_pin_name, input_wire)
                            input_pins.append(input_pin_name)

                        # --- ✨ Dynamically update the library ---
                        # (Only if we haven't seen this gate type before)
                        if cell_type not in CELL_LIBRARY:
                            # print(f"  ⓘ Info: New gate '{cell_type}' with {len(input_pins)} inputs found.")
                            CELL_LIBRARY[cell_type] = ([output_pin_name], input_pins)

                    netlist_obj.add_gate(gate_obj)
                    # --- ✨✨✨ End Dynamic Parser Logic ✨✨✨ ---
                    continue
        if netlist_obj and len(netlist_obj.gates) > 0:
            return netlist_obj
        else:
            return None
    except Exception as e:
        print(f"\n[Error] Failed to parse {netlist_file_path}: {e}");
        return None


# ##################################################################
# ########## Copied from phase1_graph_utils.py (Enhanced) #############
# ##################################################################
def convert_to_pin_graph(netlist: Netlist) -> nx.DiGraph:
    """
    (Enhanced)
    Uses CELL_LIBRARY (now dynamic) to determine pin direction.
    """
    G = nx.DiGraph();
    wire_to_pins_map = {}
    for port in netlist.inputs:
        G.add_node(port, type='Port_Input')
        wire_to_pins_map[port] = {'source_pin': port, 'sinks': []}
    for port in netlist.outputs:
        G.add_node(port, type='Port_Output')
        if port not in wire_to_pins_map: wire_to_pins_map[port] = {'source_pin': None, 'sinks': []}
        wire_to_pins_map[port]['sinks'].append(port)

    for gate_name, gate_obj in netlist.gates.items():
        cell_type = gate_obj.cell_type
        G.add_node(gate_name, type='Cell', cell_type=cell_type, is_trojan=False)

        # --- ✨✨✨ Start Enhanced Graph Logic ✨✨✨ ---
        # Retrieve pin definitions from library (now dynamic)
        output_pins, input_pins = CELL_LIBRARY.get(cell_type, ([], []))

        if not output_pins and not input_pins:
            # This shouldn't happen as parser should have added it
            # but for safety, skip unknown gates
            print(f"  ⚠️ Warning: Gate '{gate_name}' ({cell_type}) not defined in library. Ignored.")
            continue

        for port, wire in gate_obj.connections.items():
            pin_name = f"{gate_name}___{port}"

            if port in output_pins:
                G.add_node(pin_name, type='Pin_Output');
                G.add_edge(gate_name, pin_name)
                if wire not in wire_to_pins_map: wire_to_pins_map[wire] = {'source_pin': None, 'sinks': []}
                wire_to_pins_map[wire]['source_pin'] = pin_name
            elif port in input_pins:
                G.add_node(pin_name, type='Pin_Input');
                G.add_edge(pin_name, gate_name)
                if wire not in wire_to_pins_map: wire_to_pins_map[wire] = {'source_pin': None, 'sinks': []}
                wire_to_pins_map[wire]['sinks'].append(pin_name)
        # --- ✨✨✨ End Enhanced Graph Logic ✨✨✨ ---

    for wire, pins in wire_to_pins_map.items():
        source_pin = pins['source_pin']
        if source_pin:
            for sink_pin in pins['sinks']:
                if G.has_node(source_pin) and G.has_node(sink_pin):
                    G.add_edge(source_pin, sink_pin)
    return G


# --- (Functions _recursion, generate_netlist_blocks, _find_all_root_to_leaf_paths) ---
# --- (These functions are copied unchanged) ---
def _recursion(G: nx.DiGraph, current_cell: str, remaining_depth: int, max_depth: int, Direction: str) -> List[Dict]:
    if remaining_depth <= 0: return []
    current_logic_level = (max_depth - remaining_depth) + 1;
    found_nets = []
    if Direction == 'I':
        try:
            input_pins = list(G.predecessors(current_cell))
            for vp in input_pins:
                source_pins = list(G.predecessors(vp))
                for vp_prime in source_pins:
                    next_cells = list(G.predecessors(vp_prime))
                    for vc_prime in next_cells:
                        if G.nodes[vc_prime].get('type') != 'Cell': continue
                        net_data = [vc_prime, vp_prime, vp, current_cell, current_logic_level]
                        net_info = {'net': net_data,
                                    'children': _recursion(G, vc_prime, remaining_depth - 1, max_depth, 'I')}
                        found_nets.append(net_info)
        except nx.NetworkXError:
            pass
    elif Direction == 'O':
        try:
            output_pins = list(G.successors(current_cell))
            for vp in output_pins:
                sink_pins = list(G.successors(vp))
                for vp_prime in sink_pins:
                    next_cells = list(G.successors(vp_prime))
                    for vc_prime in next_cells:
                        if G.nodes[vc_prime].get('type') != 'Cell': continue
                        net_data = [vc_prime, vp_prime, vp, current_cell, current_logic_level]
                        net_info = {'net': net_data,
                                    'children': _recursion(G, vc_prime, remaining_depth - 1, max_depth, 'O')}
                        found_nets.append(net_info)
        except nx.NetworkXError:
            pass
    return found_nets


def generate_netlist_blocks(pin_graph: nx.DiGraph, logic_level: int = 4) -> Dict:
    all_blocks = {}
    cell_nodes = [n for n, d in pin_graph.nodes(data=True) if d.get('type') == 'Cell']
    for vc in tqdm(cell_nodes, desc="  (2/3) 🧱 Generating Blocks (Alg 1)", unit="gate"):
        block_tree = {
            'I': _recursion(pin_graph, vc, logic_level, logic_level, 'I'),
            'O': _recursion(pin_graph, vc, logic_level, logic_level, 'O')
        }
        all_blocks[vc] = all_blocks.get(vc, {})
        all_blocks[vc].update(block_tree)
    return all_blocks


def _find_all_root_to_leaf_paths(node_list: List[Dict]) -> List[List[List[Any]]]:
    all_paths = []

    def dfs(node: Dict, current_path: List[List[Any]]):
        current_path.append(node['net'])
        if not node['children']:
            all_paths.append(list(current_path))
        else:
            for child in node['children']: dfs(child, current_path)
        current_path.pop()

    for root_node in node_list: dfs(root_node, [])
    if not all_paths and node_list:
        for root_node in node_list:
            all_paths.append([root_node['net']])
    return all_paths


def extract_pcp_traces(netlist_blocks: Dict) -> Dict[str, List[List[str]]]:
    all_traces_map = {}

    def create_pcp_word(v_in_p: str, v_c: str, v_out_p: str) -> str:
        # Extract short pin names
        v_in_p_short = v_in_p.split('___')[-1]
        v_out_p_short = v_out_p.split('___')[-1]

        # Get cell type
        v_c_type = "Unknown"
        if netlist_obj and v_c in netlist_obj.gates:
            v_c_type = netlist_obj.gates[v_c].cell_type

        # --- ✨ (New) Cell name normalization ---
        # "Word" should match words the model was trained on
        # We map all 'and', 'and2', 'and3', 'and4' to 'and_base'
        if 'and' in v_c_type: v_c_type = 'and_base'
        if 'or' in v_c_type: v_c_type = 'or_base'
        if 'nand' in v_c_type: v_c_type = 'nand_base'
        if 'nor' in v_c_type: v_c_type = 'nor_base'
        if 'xor' in v_c_type: v_c_type = 'xor_base'
        if 'inv' in v_c_type or 'not' in v_c_type: v_c_type = 'inv_base'

        return f"{v_in_p_short}___{v_c_type}___{v_out_p_short}"

    global netlist_obj

    for center_gate, block in tqdm(netlist_blocks.items(), desc="  (3/3) 💬 Extracting Traces (Alg 2)", unit="gate"):
        input_paths = _find_all_root_to_leaf_paths(block.get('I', []))
        output_paths = _find_all_root_to_leaf_paths(block.get('O', []))

        if not input_paths: input_paths = [[['DUMMY_CELL_I', 'DUMMY_PIN_I', 'DUMMY_PIN_I', center_gate, 1]]]
        if not output_paths: output_paths = [[['DUMMY_CELL_O', 'DUMMY_PIN_O', 'DUMMY_PIN_O', center_gate, 1]]]

        generated_traces_for_gate = []
        for path_I in input_paths:
            for path_O in output_paths:
                pcp_trace_words = []
                for i in range(len(path_I) - 1, 0, -1):
                    net_a, net_b = path_I[i - 1], path_I[i]
                    word = create_pcp_word(net_b[2], net_b[3], net_a[1])
                    pcp_trace_words.append(word)
                net_a, net_b = path_I[0], path_O[0]
                center_word = create_pcp_word(net_a[2], net_a[3], net_b[2])
                pcp_trace_words.append(center_word)
                for i in range(len(path_O) - 1):
                    net_a, net_b = path_O[i], path_O[i + 1]
                    word = create_pcp_word(net_a[2], net_a[3], net_b[1])
                    pcp_trace_words.append(word)
                generated_traces_for_gate.append(pcp_trace_words)
        all_traces_map[center_gate] = generated_traces_for_gate
    return all_traces_map


# ##################################################################
# ########## Dataset Class (unchanged) #############################
# ##################################################################
class InferenceDataset(Dataset):
    def __init__(self, traces_dict: Dict[str, List[List[str]]], embeddings: KeyedVectors):
        self.embeddings = embeddings
        self.zero_vector = np.zeros(EMBEDDING_DIM).astype(np.float32)
        self.all_traces_list = []
        for gate, traces in traces_dict.items():
            for trace in traces:
                self.all_traces_list.append({'gate': gate, 'trace': trace})

    def __len__(self):
        return len(self.all_traces_list)

    def __getitem__(self, idx):
        item = self.all_traces_list[idx]
        trace_words = item['trace'];
        gate = item['gate']
        trace_tensor_data = np.zeros((MAX_TRACE_LENGTH, EMBEDDING_DIM), dtype=np.float32)
        for i, word in enumerate(trace_words):
            if i >= MAX_TRACE_LENGTH: break
            if word in self.embeddings:
                trace_tensor_data[i] = self.embeddings[word]
        return {"trace_tensor": torch.tensor(trace_tensor_data, dtype=torch.float32), "gate": gate}


# ##################################################################
# ########## Main Pipeline Execution Function (unchanged) ###################
# ##################################################################
netlist_obj: Optional[Netlist] = None


def main():
    global netlist_obj
    parser = argparse.ArgumentParser(description="Hardware Trojan Detection Tool Based on LSTM")
    parser.add_argument("verilog_file", help="Path to .v netlist file for scanning")
    parser.add_argument("--model", default=DEFAULT_MODEL_FILE, help="Path to trained .pth model file")
    parser.add_argument("--vectors", default=DEFAULT_VECTORS_FILE, help="Path to net2vec.vectors file")
    args = parser.parse_args()
    if not (os.path.exists(args.verilog_file) and os.path.exists(args.model) and os.path.exists(args.vectors)):
        print("❌ Error: One of the input files (verilog, model, or vectors) not found.")
        return
    start_time = time.time()
    print(f"--- 🔬 Phase 1: Processing {os.path.basename(args.verilog_file)} ---")
    print("  (1/3) 📄 Parsing Netlist...")
    netlist_obj = parse_netlist(args.verilog_file)
    if netlist_obj is None:
        print("❌ Processing stopped: Netlist parsing failed.")
        return
    pin_graph = convert_to_pin_graph(netlist_obj)
    net_blocks = generate_netlist_blocks(pin_graph, LOGIC_LEVEL)
    all_traces_dict = extract_pcp_traces(net_blocks)
    total_traces = sum(len(t) for t in all_traces_dict.values())
    if total_traces == 0:
        print(f"  (Parsed gates count: {len(netlist_obj.gates)})")
        print(f"  (Graph nodes count: {pin_graph.number_of_nodes()})")
        print(f"  (Graph edges count: {pin_graph.number_of_edges()})")
        print(f"  (Generated blocks count: {len(net_blocks)})")
        print("❌ Processing stopped: No valid traces extracted.")
        return
    print(f"✅ Phase 1 completed. {total_traces:,} traces extracted from {len(netlist_obj.gates):,} gates.")
    print("\n--- 🧠 Phase 3: Loading models ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  (Using device: {device})")
    try:
        embeddings = KeyedVectors.load_word2vec_format(args.vectors)
        print("  ✅ Net2Vec dictionary (vectors) loaded.")
        model = TrojanLSTM().to(device)
        model.load_state_dict(torch.load(args.model))
        model.eval()
        print(f"  ✅ Trojan Detector model ({args.model}) loaded.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    print("\n--- 🤖 Phase 4: Running Inference and Voting ---")
    inference_dataset = InferenceDataset(all_traces_dict, embeddings)
    inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    gate_votes = defaultdict(lambda: {'ht_votes': 0, 'normal_votes': 0})
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="  (1/2) 🧠 Inference running"):
            traces = batch['trace_tensor'].to(device)
            gates = batch['gate']
            outputs = model(traces)
            _, preds = torch.max(outputs, 1)
            preds_cpu = preds.cpu().numpy()
            for i in range(len(gates)):
                gate_name = gates[i]
                if preds_cpu[i] == 1:
                    gate_votes[gate_name]['ht_votes'] += 1
                else:
                    gate_votes[gate_name]['normal_votes'] += 1
    suspicious_gates = []
    for gate_name, data in tqdm(gate_votes.items(), desc="  (2/2) 🗳️ Voting process"):
        if data['ht_votes'] > data['normal_votes']:
            suspicious_gates.append(gate_name)
        elif data['ht_votes'] == data['normal_votes'] and data['ht_votes'] > 0:
            suspicious_gates.append(gate_name)
    print("\n" + "=" * 50)
    print("🏁 Scan complete")
    print("=" * 50)
    if not suspicious_gates:
        print("  ✅ Result: No hardware trojans found in this file.")
    else:
        print(f"  🚨 Warning: {len(suspicious_gates)} trojan-suspected gates found!")
        print("-" * 50)
        print("  List of suspicious gates:")
        for i, gate in enumerate(suspicious_gates):
            print(f"    {i + 1}. {gate}")
            if i > 20:
                print(f"    ... and {len(suspicious_gates) - i - 1} more gates.")
                break
    total_time = time.time() - start_time
    print(f"\n⏱️ Total scan time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()