import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import os
import re
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import numpy as np
import networkx as nx
from collections import defaultdict

# ==============================================================================
# ⚙️ Settings and Backend (same smart logic as before)
# ==============================================================================
DEFAULT_MODEL_FILE = "../Model/trojan_detector_final.pth"
DEFAULT_VECTORS_FILE = "../Model/net2vec.vectors"
LOGIC_LEVEL = 4
MAX_TRACE_LENGTH = (2 * LOGIC_LEVEL) - 1
EMBEDDING_DIM = 100
BATCH_SIZE = 128

# Global variables to hold user settings
CUSTOM_GATE_DEFINITIONS = {}
netlist_obj_global = None

# Pin mapping
PIN_NORMALIZATION = {
    'Q': 'OUT', 'QN': 'OUT', 'Q_N': 'OUT', 'Y': 'OUT', 'Z': 'OUT', 'O': 'OUT', 'SO': 'OUT',
    'A': 'IN', 'B': 'IN', 'C': 'IN', 'D': 'IN', 'E': 'IN',
    'DIN': 'IN', 'DIN1': 'IN', 'DIN2': 'IN', 'DIN3': 'IN', 'DIN4': 'IN',
    'SI': 'IN', 'D0': 'IN', 'D1': 'IN',
    'CLK': 'CLK', 'CK': 'CLK', 'RST': 'RST', 'S': 'SEL', 'SEL': 'SEL', 'EB': 'EN'
}

IGNORED_SUFFIXES = [r's\d+$', r'd\d+$', r'x\d+$', r'_[\d\w]+$', r'\d+$']
TYPE_MAPPING = {
    'nnd': 'nand', 'inv': 'not', 'buf': 'buffer', 'dff': 'dff',
    'sdff': 'dff', 'xnr': 'xnor', 'hi1': 'tiehi', 'lo1': 'tielo',
    'assign': 'buffer', 'ib1': 'buffer', 'i1': 'buffer'
}

KNOWN_OUTPUTS = {'Y', 'Q', 'QN', 'Q_N', 'Z', 'ZN', 'O', 'OUT', 'SO', 'CO', 'S', 'SUM', 'RESULT', 'R'}
KNOWN_INPUTS_PREFIXES = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'IN', 'SEL', 'CLK', 'CK', 'RST', 'EN', 'TE',
                         'ADDR', 'DATA', 'SIN', 'SCAN')


# --- Backend Classes ---

class Gate:
    def __init__(self, instance_name, cell_type):
        self.instance_name = instance_name
        self.original_type = cell_type
        self.base_type = self._clean_type(cell_type)
        self.connections = {}
        self.output_pins = []
        self.input_pins = []

    def _clean_type(self, raw_type):
        lower_type = raw_type.lower()
        clean = lower_type
        for pattern in IGNORED_SUFFIXES:
            clean = re.sub(pattern, '', clean)
        if clean in TYPE_MAPPING: return TYPE_MAPPING[clean]
        for key, val in TYPE_MAPPING.items():
            if clean.startswith(key): return val
        return clean

    def infer_pin_directions(self):
        if self.original_type in CUSTOM_GATE_DEFINITIONS:
            out_def, in_def = CUSTOM_GATE_DEFINITIONS[self.original_type]
            for port in self.connections:
                if port in out_def:
                    self.output_pins.append(port)
                else:
                    self.input_pins.append(port)
            return

        temp_inputs = []
        temp_outputs = []
        unknowns = []
        for port in self.connections.keys():
            port_upper = port.upper()
            if port_upper in KNOWN_OUTPUTS:
                temp_outputs.append(port)
                continue
            is_input = False
            for prefix in KNOWN_INPUTS_PREFIXES:
                if port_upper.startswith(prefix):
                    if prefix == 'S' and ('SUM' in port_upper):
                        pass
                    else:
                        temp_inputs.append(port)
                        is_input = True
                        break
            if is_input: continue
            unknowns.append(port)

        if not temp_outputs:
            if unknowns:
                temp_outputs.append(unknowns[0])
                temp_inputs.extend(unknowns[1:])
            elif temp_inputs:
                temp_outputs.append(temp_inputs.pop())
        else:
            temp_inputs.extend(unknowns)
        self.output_pins = temp_outputs
        self.input_pins = temp_inputs


class Netlist:
    def __init__(self, module_name):
        self.module_name = module_name
        self.inputs = set();
        self.outputs = set();
        self.wires = set();
        self.gates = {}

    def add_gate(self, gate_obj): self.gates[gate_obj.instance_name] = gate_obj


class TrojanLSTM(nn.Module):
    def __init__(self, input_size=100, hidden_size=128, num_layers=2, output_size=2):
        super(TrojanLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 128).to(x.device)
        c0 = torch.zeros(2, x.size(0), 128).to(x.device)
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        return self.fc(h_n[-1])


# --- Processing functions (simplified for GUI) ---

def parse_netlist_logic(filepath):
    re_module = re.compile(r'^\s*module\s+([\w\d_]+)', re.IGNORECASE)
    re_port = re.compile(r'^\s*(input|output)\s+(.+);', re.IGNORECASE)
    re_assign = re.compile(r'^\s*assign\s+([\w\d_\[\]:]+)\s*=\s*([\w\d_\[\]:\'b]+);', re.IGNORECASE)
    re_gate = re.compile(r'^\s*([\w\d_]+)\s+([\w\d_:\\]+)\s*\((.*?)\);', re.DOTALL)
    re_port_conn = re.compile(r'\.([\w\d_]+)\s*\(([\w\d_\[\]:\'b]*)\)')

    netlist = None
    assign_counter = 0

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = re.sub(r'//.*', '', f.read())
            statements = content.split(';')

            for stmt in statements:
                stmt = stmt.strip()
                if not stmt: continue

                if netlist is None:
                    match = re_module.search(stmt)
                    if match: netlist = Netlist(match.group(1))
                    continue

                match = re_port.search(stmt + ';')
                if match:
                    ptype, pstr = match.groups()
                    ports = [p.strip() for p in pstr.split(',') if p.strip()]
                    if ptype.lower() == 'input':
                        netlist.inputs.update(ports)
                    else:
                        netlist.outputs.update(ports)
                    continue

                match = re_assign.search(stmt + ';')
                if match:
                    dest, src = match.groups()
                    g = Gate(f"assign_{assign_counter}", 'buffer')
                    assign_counter += 1
                    g.connections['Q'] = dest;
                    g.connections['A'] = src
                    g.infer_pin_directions()
                    netlist.add_gate(g)
                    continue

                match = re_gate.search(stmt + ';')
                if match:
                    ctype, name, conn_str = match.groups()
                    name = name.replace('\\', '').strip()
                    g = Gate(name, ctype)
                    for pm in re_port_conn.finditer(conn_str):
                        if pm.group(2): g.connections[pm.group(1)] = pm.group(2)
                    g.infer_pin_directions()
                    netlist.add_gate(g)
                    continue
        return netlist
    except Exception as e:
        return None


def build_graph(netlist):
    G = nx.DiGraph()
    wire_map = {}
    for p in netlist.inputs:
        G.add_node(p, type='Port_Input');
        wire_map[p] = {'src': p, 'dst': []}
    for p in netlist.outputs:
        G.add_node(p, type='Port_Output')
        if p not in wire_map: wire_map[p] = {'src': None, 'dst': []}
        wire_map[p]['dst'].append(p)

    for name, g in netlist.gates.items():
        G.add_node(name, type='Cell', cell_type=g.base_type)
        for p in g.output_pins:
            w = g.connections[p]
            pname = f"{name}___{p}"
            G.add_node(pname, type='Pin_Output');
            G.add_edge(name, pname)
            if w not in wire_map: wire_map[w] = {'src': None, 'dst': []}
            wire_map[w]['src'] = pname
        for p in g.input_pins:
            w = g.connections[p]
            pname = f"{name}___{p}"
            G.add_node(pname, type='Pin_Input');
            G.add_edge(pname, name)
            if w not in wire_map: wire_map[w] = {'src': None, 'dst': []}
            wire_map[w]['dst'].append(pname)

    for w, d in wire_map.items():
        if d['src']:
            for dst in d['dst']:
                if G.has_node(d['src']) and G.has_node(dst): G.add_edge(d['src'], dst)
    return G


def get_norm_pin(raw):
    p = raw.split('___')[-1].upper()
    if p in PIN_NORMALIZATION: return PIN_NORMALIZATION[p]
    for k, v in PIN_NORMALIZATION.items():
        if p.startswith(k): return v
    if p.startswith(('I', 'A', 'D')): return 'IN'
    return 'OUT'


def extract_traces(G):
    blocks = {}
    cells = [n for n, d in G.nodes(data=True) if d.get('type') == 'Cell']

    def recurs(curr, depth, direction):
        if depth <= 0: return []
        res = []
        if direction == 'I':
            try:
                for vp in G.predecessors(curr):
                    for vpp in G.predecessors(vp):
                        for vnext in G.predecessors(vpp):
                            if G.nodes[vnext].get('type') != 'Cell': continue
                            ctype = G.nodes[vnext].get('cell_type', 'unk')
                            res.append(
                                {'net': [vnext, vpp, vp, curr, depth, ctype], 'ch': recurs(vnext, depth - 1, 'I')})
            except:
                pass
        else:
            try:
                for vp in G.successors(curr):
                    for vpp in G.successors(vp):
                        for vnext in G.successors(vpp):
                            if G.nodes[vnext].get('type') != 'Cell': continue
                            ctype = G.nodes[vnext].get('cell_type', 'unk')
                            res.append(
                                {'net': [vnext, vpp, vp, curr, depth, ctype], 'ch': recurs(vnext, depth - 1, 'O')})
            except:
                pass
        return res

    for c in cells:
        blocks[c] = {'I': recurs(c, LOGIC_LEVEL, 'I'), 'O': recurs(c, LOGIC_LEVEL, 'O'),
                     'ct': G.nodes[c].get('cell_type')}

    traces_map = {}

    def get_paths(nodes):
        paths = []

        def dfs(n, p):
            p.append(n['net'])
            if not n['ch']:
                paths.append(list(p))
            else:
                for ch in n['ch']: dfs(ch, p)
            p.pop()

        for r in nodes: dfs(r, [])
        if not paths and nodes:
            for r in nodes: paths.append([r['net']])
        return paths

    def mk_word(p1, ct, p2):
        return f"{get_norm_pin(p1)}___{ct}___{get_norm_pin(p2)}"

    for c, b in blocks.items():
        pi = get_paths(b['I'])
        po = get_paths(b['O'])
        if not pi: pi = [[['DUMMY', 'IN', 'IN', c, 1, 'dummy']]]
        if not po: po = [[['DUMMY', 'OUT', 'OUT', c, 1, 'dummy']]]

        t_list = []
        for p_in in pi:
            for p_out in po:
                t = []
                for i in range(len(p_in) - 1, 0, -1):
                    t.append(mk_word(p_in[i][2], p_in[i][5], p_in[i - 1][1]))
                t.append(mk_word(p_in[0][2], b['ct'], p_out[0][2]))
                for i in range(len(p_out) - 1):
                    t.append(mk_word(p_out[i][2], p_out[i][5], p_out[i + 1][1]))
                t_list.append(t)
        traces_map[c] = t_list
    return traces_map


# ==============================================================================
# 🖥️ Graphical User Interface (GUI)
# ==============================================================================

class TrojanDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Hardware Trojan Detector")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        # --- Header ---
        header_frame = tk.Frame(root, bg="#2c3e50", pady=10)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="Verilog Netlist Trojan Detection System",
                 font=("Segoe UI", 16, "bold"), fg="white", bg="#2c3e50").pack()

        # --- File Selection ---
        file_frame = tk.LabelFrame(root, text="Netlist File Selection (.v)", font=("Segoe UI", 10, "bold"), bg="#f0f0f0",
                                   padx=10, pady=10)
        file_frame.pack(fill="x", padx=20, pady=10)

        self.file_path_var = tk.StringVar()
        self.entry_file = tk.Entry(file_frame, textvariable=self.file_path_var, width=80, font=("Consolas", 10))
        self.entry_file.pack(side="left", padx=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_file, bg="#3498db", fg="white",
                  font=("Segoe UI", 9, "bold")).pack(side="left")

        # --- Manual Config ---
        config_frame = tk.LabelFrame(root, text="Manual Gate Definition (Optional - for custom naming)",
                                     font=("Segoe UI", 10, "bold"), bg="#f0f0f0", padx=10, pady=10)
        config_frame.pack(fill="x", padx=20, pady=5)

        tk.Label(config_frame, text="Gate Name (e.g., nor2s):", bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.entry_gate_name = tk.Entry(config_frame, width=15)
        self.entry_gate_name.grid(row=0, column=1, padx=5)

        tk.Label(config_frame, text="Outputs (comma-separated):", bg="#f0f0f0").grid(row=0, column=2, padx=5)
        self.entry_gate_outs = tk.Entry(config_frame, width=20)
        self.entry_gate_outs.grid(row=0, column=3, padx=5)

        tk.Label(config_frame, text="Inputs:", bg="#f0f0f0").grid(row=0, column=4, padx=5)
        self.entry_gate_ins = tk.Entry(config_frame, width=20)
        self.entry_gate_ins.grid(row=0, column=5, padx=5)

        tk.Button(config_frame, text="Add", command=self.add_custom_gate, bg="#27ae60", fg="white").grid(row=0,
                                                                                                            column=6,
                                                                                                            padx=10)

        self.listbox_custom = tk.Listbox(config_frame, height=3, width=100, font=("Consolas", 9))
        self.listbox_custom.grid(row=1, column=0, columnspan=7, pady=5)

        # --- Action ---
        action_frame = tk.Frame(root, bg="#f0f0f0", pady=10)
        action_frame.pack(fill="x")
        self.btn_scan = tk.Button(action_frame, text="🚀 Start Scan & Analysis", command=self.start_scan_thread,
                                  bg="#e74c3c", fg="white", font=("Segoe UI", 12, "bold"), padx=20, pady=5)
        self.btn_scan.pack()

        # --- Log Output ---
        log_frame = tk.LabelFrame(root, text="Analysis Report (Logs)", font=("Segoe UI", 10, "bold"), bg="#f0f0f0", padx=10,
                                  pady=10)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', font=("Consolas", 10), bg="#ecf0f1")
        self.log_text.pack(fill="both", expand=True)

        # Color tags for logging
        self.log_text.tag_config("INFO", foreground="black")
        self.log_text.tag_config("SUCCESS", foreground="green")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("WARN", foreground="#d35400")
        self.log_text.tag_config("RESULT", foreground="blue", font=("Consolas", 11, "bold"))

    def log(self, message, level="INFO"):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n", level)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Verilog Files", "*.v"), ("All Files", "*.*")])
        if filename:
            self.file_path_var.set(filename)

    def add_custom_gate(self):
        name = self.entry_gate_name.get().strip()
        outs = [x.strip() for x in self.entry_gate_outs.get().split(',') if x.strip()]
        ins = [x.strip() for x in self.entry_gate_ins.get().split(',') if x.strip()]

        if not name or not outs:
            messagebox.showwarning("Error", "Gate name and at least one output are required.")
            return

        CUSTOM_GATE_DEFINITIONS[name] = (outs, ins)
        self.listbox_custom.insert(tk.END, f"Gate: {name} | Out: {outs} | In: {ins}")
        self.entry_gate_name.delete(0, tk.END)
        self.entry_gate_outs.delete(0, tk.END)
        self.entry_gate_ins.delete(0, tk.END)
        self.log(f"Manual configuration added: {name}", "INFO")

    def start_scan_thread(self):
        filepath = self.file_path_var.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Error", "Please select a valid file.")
            return

        if not os.path.exists(DEFAULT_MODEL_FILE) or not os.path.exists(DEFAULT_VECTORS_FILE):
            messagebox.showerror("Error", f"Model files not found.\nPlease check the Model folder exists.")
            return

        self.btn_scan.config(state="disabled", text="⏳ Analyzing...")
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

        # Run scan in separate thread so GUI doesn't freeze
        thread = threading.Thread(target=self.run_scan_logic, args=(filepath,))
        thread.start()

    def run_scan_logic(self, filepath):
        global netlist_obj_global
        start_t = time.time()

        try:
            self.log(f"--- Starting Processing: {os.path.basename(filepath)} ---", "INFO")

            # 1. Parsing
            self.log("1. Parsing Netlist...", "INFO")
            netlist_obj_global = parse_netlist_logic(filepath)
            if not netlist_obj_global:
                self.log("❌ Error parsing file.", "ERROR")
                return

            gate_count = len(netlist_obj_global.gates)
            self.log(f"✅ {gate_count} gates identified.", "SUCCESS")

            # 2. Graph
            self.log("2. Building Connection Graph...", "INFO")
            G = build_graph(netlist_obj_global)
            self.log(f"✅ Graph created (Nodes: {G.number_of_nodes()})", "SUCCESS")

            # 3. Traces
            self.log("3. Extracting Traces...", "INFO")
            traces = extract_traces(G)
            total_traces = sum(len(t) for t in traces.values())

            if total_traces == 0:
                self.log("⚠️ No traces extracted (circuit too small?)", "WARN")
                return

            self.log(f"✅ {total_traces:,} traces extracted.", "SUCCESS")

            # 4. Load Model
            self.log("4. Loading AI Model...", "INFO")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log(f"   (Engine: {device})", "INFO")

            embeddings = KeyedVectors.load_word2vec_format(DEFAULT_VECTORS_FILE)
            model = TrojanLSTM().to(device)
            model.load_state_dict(torch.load(DEFAULT_MODEL_FILE, map_location=device))
            model.eval()

            # 5. Inference
            self.log("5. Running Inference...", "INFO")

            # Prepare data for model
            all_data_list = []
            for gname, tlist in traces.items():
                for t in tlist: all_data_list.append({'g': gname, 't': t})

            # Batch processing (simplified without heavy DataLoader for GUI)
            gate_votes = defaultdict(lambda: {'ht': 0, 'n': 0})

            # To prevent freezing, do manual batching
            total_items = len(all_data_list)
            processed = 0

            for i in range(0, total_items, BATCH_SIZE):
                batch = all_data_list[i:i + BATCH_SIZE]

                # Convert to tensor
                tensor_batch = np.zeros((len(batch), MAX_TRACE_LENGTH, EMBEDDING_DIM), dtype=np.float32)
                gates_in_batch = []

                for idx, item in enumerate(batch):
                    gates_in_batch.append(item['g'])
                    for w_idx, word in enumerate(item['t']):
                        if w_idx >= MAX_TRACE_LENGTH: break

                        # Fuzzy match logic
                        vec = None
                        if word in embeddings:
                            vec = embeddings[word]
                        else:
                            parts = word.split('___')
                            if len(parts) == 3 and parts[1] in embeddings: vec = embeddings[parts[1]]

                        if vec is not None: tensor_batch[idx, w_idx] = vec

                t_in = torch.tensor(tensor_batch).to(device)
                with torch.no_grad():
                    out = model(t_in)
                    _, preds = torch.max(out, 1)
                    preds = preds.cpu().numpy()

                for j, p in enumerate(preds):
                    if p == 1:
                        gate_votes[gates_in_batch[j]]['ht'] += 1
                    else:
                        gate_votes[gates_in_batch[j]]['n'] += 1

                processed += len(batch)
                if processed % 1000 == 0:
                    self.log(f"   ... Processing {processed}/{total_items}", "INFO")

            # 6. Results
            suspicious = [g for g, v in gate_votes.items() if v['ht'] > v['n']]

            self.log("\n" + "=" * 40, "RESULT")
            if suspicious:
                self.log(f"🚨 Warning: Trojan Detected!", "ERROR")
                self.log(f"Suspicious gate count: {len(suspicious)}", "ERROR")
                self.log(f"Sample gates: {suspicious[:10]}", "ERROR")

                # Save
                with open("gui_suspicious_report.txt", "w") as f:
                    f.write("\n".join(suspicious))
                self.log("📄 Full list saved to gui_suspicious_report.txt.", "INFO")
            else:
                self.log("✅ Circuit is clean. No trojans found.", "SUCCESS")

            elapsed = time.time() - start_t
            self.log(f"\n⏱️ Total time: {elapsed:.2f} seconds", "RESULT")

        except Exception as e:
            self.log(f"❌ Unexpected error: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()

        finally:
            self.btn_scan.config(state="normal", text="🚀 Start Scan & Analysis")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrojanDetectorApp(root)
    root.mainloop()