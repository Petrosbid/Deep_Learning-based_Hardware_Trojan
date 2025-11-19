#
# detector.py
# (نسخه نهایی و تعاملی: تشخیص هوشمند + قابلیت تعریف دستی توسط کاربر)
#
import os
import re
import time
import argparse
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Optional, Any

# --- تنظیمات ---
DEFAULT_MODEL_FILE = "../Model/trojan_detector_final.pth"
DEFAULT_VECTORS_FILE = "../Model/net2vec.vectors"
LOGIC_LEVEL = 4
MAX_TRACE_LENGTH = (2 * LOGIC_LEVEL) - 1
EMBEDDING_DIM = 100
BATCH_SIZE = 128
NUM_WORKERS = 0

# ==============================================================================
# 🔧 مخزن تعاریف (اینجا با ورودی کاربر آپدیت می‌شود)
# ==============================================================================
# فرمت: 'gate_name': (['OUTPUTS'], ['INPUTS'])
CUSTOM_GATE_DEFINITIONS = {}

# لیست سفید خروجی‌ها (استاندارد جهانی)
KNOWN_OUTPUTS = {
    'Y', 'Q', 'QN', 'Q_N', 'Z', 'ZN', 'O', 'OUT', 'SO', 'CO', 'S', 'SUM', 'RESULT', 'R',
    'Q_REG', 'EQ', 'LT', 'GT'
}

# لیست پیشوندهای ورودی
KNOWN_INPUTS_PREFIXES = (
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'IN', 'SEL', 'CLK', 'CK', 'RST', 'RESET', 'EN', 'TE', 'TI', 'ADDR', 'DATA',
    'SIN', 'SCAN'
)

IGNORED_SUFFIXES = [r's\d+', r'd\d+', r'x\d+', r'_[\d\w]+', r'\d+']
TYPE_MAPPING = {
    'nnd': 'nand', 'inv': 'not', 'buf': 'buffer',
    'dff': 'dff', 'sdff': 'dff', 'xnr': 'xnor'
}


# ##################################################################
# ########## رابط کاربری تعاملی (جدید) ############################
# ##################################################################
def run_interactive_setup():
    """
    این تابع قبل از شروع پردازش اجرا می‌شود و اطلاعات خاص کاربر را می‌گیرد.
    """
    print("\n" + "=" * 60)
    print(" 🛠️  بخش تنظیمات دستی پارسر (Manual Configuration)  🛠️")
    print("=" * 60)
    print("توضیح: این برنامه به صورت پیش‌فرض پین‌های استاندارد (مثل Q, Y, OUT) را خروجی")
    print("و پین‌های (A, B, I, D) را ورودی در نظر می‌گیرد.")
    print("\n⚠️  سوال: آیا در فایل شما گیتی وجود دارد که نام‌گذاری پین‌هایش برعکس یا عجیب باشد؟")
    print("    (مثلاً گیتی که پین 'A' خروجی باشد و 'Q' ورودی؟)")

    while True:
        choice = input("\n>> آیا نیاز به تعریف دستی دارید؟ (y/n): ").strip().lower()
        if choice in ['y', 'n', 'yes', 'no']:
            break
        print("خطا: لطفاً فقط 'y' یا 'n' وارد کنید.")

    if choice in ['n', 'no']:
        print("✅ بسیار عالی. از تنظیمات هوشمند پیش‌فرض استفاده می‌شود.")
        return

    print("\n--- 📝 ورود اطلاعات گیت‌های خاص ---")
    print("نکته: نام گیت را دقیقاً همانطور که در فایل Verilog است وارد کنید (مثلاً 'nor2s3').")

    while True:
        print("-" * 30)
        gate_name = input("1. نام نوع گیت (Gate Type): ").strip()
        if not gate_name:
            print("نام گیت نمی‌تواند خالی باشد.")
            continue

        out_str = input(f"2. لیست پین‌های خروجی '{gate_name}' (با کاما , جدا کنید): ").strip()
        outs = [x.strip() for x in out_str.split(',') if x.strip()]

        in_str = input(f"3. لیست پین‌های ورودی '{gate_name}' (با کاما , جدا کنید): ").strip()
        ins = [x.strip() for x in in_str.split(',') if x.strip()]

        if not outs:
            print("❌ خطا: هر گیت باید حداقل یک خروجی داشته باشد.")
            continue

        # ذخیره در دیکشنری گلوبال
        CUSTOM_GATE_DEFINITIONS[gate_name] = (outs, ins)
        print(f"✅ ثبت شد: [{gate_name}] -> Outputs:{outs} | Inputs:{ins}")

        more = input("\n>> آیا گیت دیگری برای اضافه کردن دارید؟ (y/n): ").strip().lower()
        if more not in ['y', 'yes']:
            break

    print("\n✅ تنظیمات دستی پایان یافت. شروع پردازش...")
    print("=" * 60 + "\n")


# ##################################################################
# ########## مدل LSTM ##############################################
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
# ########## کلاس گیت و منطق تشخیص جهت ############################
# ##################################################################
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
        for key, val in TYPE_MAPPING.items():
            if key in clean:
                return clean.replace(key, val)
        return clean

    def infer_pin_directions(self):
        # 1. اولویت اول: تنظیمات دستی کاربر
        # ما نوع اصلی (original_type) را چک می‌کنیم که کاربر وارد کرده است
        if self.original_type in CUSTOM_GATE_DEFINITIONS:
            defined_outs, defined_ins = CUSTOM_GATE_DEFINITIONS[self.original_type]
            for port in self.connections:
                # چک کردن دقیق نام پورت
                if port in defined_outs:
                    self.output_pins.append(port)
                elif port in defined_ins:
                    self.input_pins.append(port)
                else:
                    # اگر پورتی در کد بود که کاربر تعریف نکرده، طبق منطق هوشمند پیش برو
                    self._heuristic_inference(port)
            return

        # اگر در تنظیمات دستی نبود، کل پورت‌ها را بده به تشخیص هوشمند
        for port in self.connections:
            self._heuristic_inference(port)

        # بررسی نهایی برای گیت‌های بدون خروجی مشخص شده
        if not self.output_pins and self.input_pins:
            # اگر همه ورودی شدند و خروجی نداریم، این اشتباه است.
            # آخرین ورودی را برمی‌داریم و خروجی می‌کنیم (حدس آخر)
            popped = self.input_pins.pop()
            self.output_pins.append(popped)

    def _heuristic_inference(self, port):
        """منطق تشخیص هوشمند برای یک پورت"""
        port_upper = port.upper()

        # 2. خروجی‌های استاندارد
        if port_upper in KNOWN_OUTPUTS:
            self.output_pins.append(port)
            return

        # 3. ورودی‌های استاندارد
        is_input = False
        for prefix in KNOWN_INPUTS_PREFIXES:
            if port_upper.startswith(prefix):
                if prefix == 'S' and ('SUM' in port_upper):
                    pass  # S اگر SUM باشد ورودی نیست
                else:
                    self.input_pins.append(port)
                    is_input = True
                    break

        if is_input: return

        # 4. اگر ناشناخته بود
        # اگر هنوز خروجی نداریم، این را خروجی بگیر. اگر داریم، ورودی بگیر.
        if not self.output_pins:
            self.output_pins.append(port)
        else:
            self.input_pins.append(port)


class Netlist:
    def __init__(self, module_name):
        self.module_name = module_name
        self.inputs = set();
        self.outputs = set();
        self.wires = set();
        self.gates = {}

    def add_gate(self, gate_obj):
        self.gates[gate_obj.instance_name] = gate_obj


# ##################################################################
# ########## پارس کردن فایل ########################################
# ##################################################################
def parse_netlist_dynamic(netlist_file_path: str) -> Optional[Netlist]:
    re_module = re.compile(r'^\s*module\s+([\w\d_]+)', re.IGNORECASE)
    re_port = re.compile(r'^\s*(input|output)\s+(.+);', re.IGNORECASE)
    re_wire = re.compile(r'^\s*wire\s+(.+);', re.IGNORECASE)
    re_assign = re.compile(r'^\s*assign\s+([\w\d_\[\]:]+)\s*=\s*([\w\d_\[\]:\'b]+);', re.IGNORECASE)
    re_gate = re.compile(r'^\s*([\w\d_]+)\s+([\w\d_:\\]+)\s*\((.*?)\);', re.DOTALL)
    re_port_conn = re.compile(r'\.([\w\d_]+)\s*\(([\w\d_\[\]:\'b]*)\)')

    netlist_obj = None
    buffer = ""
    assign_counter = 0

    try:
        with open(netlist_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'): continue
                buffer += " " + line
                if not buffer.endswith(';'): continue

                complete_line = buffer.strip()
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

                match = re_assign.search(complete_line)
                if match:
                    dest_wire, source_wire = match.groups()
                    gate_obj = Gate(f"assign_{assign_counter}", 'buffer')
                    assign_counter += 1
                    gate_obj.connections['Q'] = dest_wire
                    gate_obj.connections['A'] = source_wire
                    gate_obj.infer_pin_directions()
                    netlist_obj.add_gate(gate_obj)
                    continue

                match = re_gate.search(complete_line)
                if match:
                    cell_type, instance_name, connections_str = match.groups()
                    instance_name = instance_name.replace('\\', '').strip()
                    gate_obj = Gate(instance_name, cell_type)

                    for port_match in re_port_conn.finditer(connections_str):
                        port_name = port_match.group(1)
                        wire_name = port_match.group(2)
                        if wire_name: gate_obj.connections[port_name] = wire_name

                    gate_obj.infer_pin_directions()
                    netlist_obj.add_gate(gate_obj)
                    continue

        if netlist_obj and len(netlist_obj.gates) > 0:
            return netlist_obj
        else:
            return None
    except Exception as e:
        print(f"\n[Error] Failed to parse: {e}")
        return None


# ##################################################################
# ########## سایر توابع (تبدیل به گراف و ...) #######################
# ##################################################################
def convert_to_pin_graph(netlist: Netlist) -> nx.DiGraph:
    G = nx.DiGraph()
    wire_to_pins_map = {}

    for port in netlist.inputs:
        G.add_node(port, type='Port_Input')
        wire_to_pins_map[port] = {'source_pin': port, 'sinks': []}
    for port in netlist.outputs:
        G.add_node(port, type='Port_Output')
        if port not in wire_to_pins_map: wire_to_pins_map[port] = {'source_pin': None, 'sinks': []}
        wire_to_pins_map[port]['sinks'].append(port)

    for gate_name, gate_obj in netlist.gates.items():
        G.add_node(gate_name, type='Cell', cell_type=gate_obj.base_type, is_trojan=False)
        for port in gate_obj.output_pins:
            wire = gate_obj.connections[port]
            pin_name = f"{gate_name}___{port}"
            G.add_node(pin_name, type='Pin_Output')
            G.add_edge(gate_name, pin_name)
            if wire not in wire_to_pins_map: wire_to_pins_map[wire] = {'source_pin': None, 'sinks': []}
            wire_to_pins_map[wire]['source_pin'] = pin_name
        for port in gate_obj.input_pins:
            wire = gate_obj.connections[port]
            pin_name = f"{gate_name}___{port}"
            G.add_node(pin_name, type='Pin_Input')
            G.add_edge(pin_name, gate_name)
            if wire not in wire_to_pins_map: wire_to_pins_map[wire] = {'source_pin': None, 'sinks': []}
            wire_to_pins_map[wire]['sinks'].append(pin_name)

    for wire, pins in wire_to_pins_map.items():
        source_pin = pins['source_pin']
        if source_pin:
            for sink_pin in pins['sinks']:
                if G.has_node(source_pin) and G.has_node(sink_pin): G.add_edge(source_pin, sink_pin)
    return G


def _recursion(G: nx.DiGraph, current_cell: str, remaining_depth: int, max_depth: int, Direction: str) -> List[Dict]:
    if remaining_depth <= 0: return []
    current_logic_level = (max_depth - remaining_depth) + 1
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
    for vc in tqdm(cell_nodes, desc="  (2/3) 🧱 Generating Blocks", unit="gate"):
        block_tree = {
            'I': _recursion(pin_graph, vc, logic_level, logic_level, 'I'),
            'O': _recursion(pin_graph, vc, logic_level, logic_level, 'O')
        }
        all_blocks[vc] = block_tree
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
        for root_node in node_list: all_paths.append([root_node['net']])
    return all_paths


def extract_pcp_traces(netlist_blocks: Dict) -> Dict[str, List[List[str]]]:
    all_traces_map = {}
    global netlist_obj_global

    def create_pcp_word(v_in_p: str, v_c_type: str, v_out_p: str) -> str:
        v_in_p_short = v_in_p.split('___')[-1]
        v_out_p_short = v_out_p.split('___')[-1]
        return f"{v_in_p_short}___{v_c_type}___{v_out_p_short}"

    for center_gate, block in tqdm(netlist_blocks.items(), desc="  (3/3) 💬 Extracting Traces", unit="gate"):
        input_paths = _find_all_root_to_leaf_paths(block.get('I', []))
        output_paths = _find_all_root_to_leaf_paths(block.get('O', []))
        if not input_paths: input_paths = [[['DUMMY', 'IN', 'IN', center_gate, 1]]]
        if not output_paths: output_paths = [[['DUMMY', 'OUT', 'OUT', center_gate, 1]]]

        generated_traces_for_gate = []
        for path_I in input_paths:
            for path_O in output_paths:
                pcp_trace_words = []
                for i in range(len(path_I) - 1, 0, -1):
                    net_a, net_b = path_I[i - 1], path_I[i]
                    cell_type = "unknown"
                    if netlist_obj_global and net_b[3] in netlist_obj_global.gates:
                        cell_type = netlist_obj_global.gates[net_b[3]].base_type
                    pcp_trace_words.append(create_pcp_word(net_b[2], cell_type, net_a[1]))

                net_a_in = path_I[0];
                net_b_out = path_O[0]
                center_type = "unknown"
                if netlist_obj_global and center_gate in netlist_obj_global.gates:
                    center_type = netlist_obj_global.gates[center_gate].base_type
                pcp_trace_words.append(create_pcp_word(net_a_in[2], center_type, net_b_out[2]))

                for i in range(len(path_O) - 1):
                    net_a, net_b = path_O[i], path_O[i + 1]
                    cell_type = "unknown"
                    if netlist_obj_global and net_a[3] in netlist_obj_global.gates:
                        cell_type = netlist_obj_global.gates[net_a[3]].base_type
                    pcp_trace_words.append(create_pcp_word(net_a[2], cell_type, net_b[1]))
                generated_traces_for_gate.append(pcp_trace_words)
        all_traces_map[center_gate] = generated_traces_for_gate
    return all_traces_map


class InferenceDataset(Dataset):
    def __init__(self, traces_dict: Dict[str, List[List[str]]], embeddings: KeyedVectors):
        self.embeddings = embeddings
        self.all_traces_list = []
        for gate, traces in traces_dict.items():
            for trace in traces: self.all_traces_list.append({'gate': gate, 'trace': trace})

    def __len__(self):
        return len(self.all_traces_list)

    def __getitem__(self, idx):
        item = self.all_traces_list[idx]
        trace_words = item['trace'];
        gate = item['gate']
        trace_tensor_data = np.zeros((MAX_TRACE_LENGTH, EMBEDDING_DIM), dtype=np.float32)
        for i, word in enumerate(trace_words):
            if i >= MAX_TRACE_LENGTH: break
            if word in self.embeddings: trace_tensor_data[i] = self.embeddings[word]
        return {"trace_tensor": torch.tensor(trace_tensor_data, dtype=torch.float32), "gate": gate}


# ##################################################################
# ########## Main ##################################################
# ##################################################################
netlist_obj_global: Optional[Netlist] = None


def main():
    global netlist_obj_global
    parser = argparse.ArgumentParser(description="Hardware Trojan Detector (Interactive)")
    parser.add_argument("verilog_file", help="Path to .v netlist")
    parser.add_argument("--model", default=DEFAULT_MODEL_FILE, help="Path to .pth model")
    parser.add_argument("--vectors", default=DEFAULT_VECTORS_FILE, help="Path to .vectors")
    args = parser.parse_args()

    if not (os.path.exists(args.verilog_file) and os.path.exists(args.model) and os.path.exists(args.vectors)):
        print("❌ خطا: یکی از فایل‌های ورودی (netlist, model, vectors) یافت نشد.")
        return

    # 1. اجرای تنظیمات تعاملی (Ask User)
    run_interactive_setup()

    start_time = time.time()
    print(f"--- 🔬 فاز 1: پردازش {os.path.basename(args.verilog_file)} ---")

    print("  (1/3) 📄 Parsing Netlist...")
    netlist_obj_global = parse_netlist_dynamic(args.verilog_file)
    if netlist_obj_global is None: return
    print(f"    ✅ {len(netlist_obj_global.gates)} گیت شناسایی شد.")

    pin_graph = convert_to_pin_graph(netlist_obj_global)
    print(f"    ✅ گراف ساخته شد (گره‌ها: {pin_graph.number_of_nodes()})")

    net_blocks = generate_netlist_blocks(pin_graph, LOGIC_LEVEL)
    all_traces_dict = extract_pcp_traces(net_blocks)

    total_traces = sum(len(t) for t in all_traces_dict.values())
    if total_traces == 0:
        print("❌ هیچ ردیابی استخراج نشد.")
        return
    print(f"✅ {total_traces:,} ردیابی استخراج شد.")

    print("\n--- 🧠 فاز 2: استنتاج هوش مصنوعی ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        embeddings = KeyedVectors.load_word2vec_format(args.vectors)
        model = TrojanLSTM().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ خطا در لود مدل: {e}")
        return

    inference_dataset = InferenceDataset(all_traces_dict, embeddings)
    inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    gate_votes = defaultdict(lambda: {'ht': 0, 'norm': 0})
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="  🧠 Inference"):
            traces = batch['trace_tensor'].to(device)
            gates = batch['gate']
            outputs = model(traces)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            for i, gate in enumerate(gates):
                if preds[i] == 1:
                    gate_votes[gate]['ht'] += 1
                else:
                    gate_votes[gate]['norm'] += 1

    suspicious_gates = [g for g, v in gate_votes.items() if v['ht'] > v['norm']]
    print("\n--- 📊 نتایج نهایی ---")
    if suspicious_gates:
        print(f"🚨 تروجان شناسایی شد! ({len(suspicious_gates)} گیت مشکوک)")
        print("نمونه گیت‌های آلوده:", suspicious_gates[:5])
        with open("suspicious_gates.txt", "w") as f:
            f.write("\n".join(suspicious_gates))
        print("📄 لیست کامل در suspicious_gates.txt ذخیره شد.")
    else:
        print("✅ مدار پاک است.")
    print(f"\n⏱️ زمان کل: {time.time() - start_time:.2f} ثانیه")


if __name__ == "__main__":
    main()