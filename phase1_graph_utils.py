#
# phase1_graph_utils.py
# (فاز 1: تبدیل به گراف پین-به-پین و استخراج ویژگی)
#
import networkx as nx
from typing import List, Dict, Set, Tuple, Any
from netlist_parser import Netlist, Gate  # وارد کردن از فایل قبلی


# ##################################################################
# ########## TODO 1: پیاده‌سازی واقعی Cell-Pin Splitter #########
# ########## (بدون تغییر) ########################################
# ##################################################################
def convert_to_pin_graph(netlist: Netlist) -> nx.DiGraph:
    """
    نت‌لیست را به یک گراف پین-به-پین واقعی (مطابق شکل 3c) تبدیل می‌کند.
    (نسخه اصلاح شده با فرض هوشمندتر برای تشخیص پین)
    """
    G = nx.DiGraph()
    wire_to_pins_map = {}

    # 1. اضافه کردن پورت‌های ورودی/خروجی ماژول
    for port in netlist.inputs:
        G.add_node(port, type='Port_Input')
        wire_to_pins_map[port] = {'source_pin': port, 'sinks': []}
    for port in netlist.outputs:
        G.add_node(port, type='Port_Output')
        if port not in wire_to_pins_map:
            wire_to_pins_map[port] = {'source_pin': None, 'sinks': []}
        wire_to_pins_map[port]['sinks'].append(port)

    # 2. اضافه کردن گره‌های Cell و Pin و یال‌های داخلی آنها
    for gate_name, gate_obj in netlist.gates.items():
        # is_trojan=False چون در حالت تشخیص هستیم
        G.add_node(gate_name, type='Cell', cell_type=gate_obj.cell_type, is_trojan=False)

        for port, wire in gate_obj.connections.items():
            pin_name = f"{gate_name}___{port}"

            # --- ✨✨✨ کد اصلاح شده ✨✨✨ ---
            # فرض هوشمندتر برای تشخیص پین‌های خروجی
            port_name_upper = port.upper()
            is_output_port = (
                    port_name_upper.startswith('Q') or  # برای فلیپ‌فلاپ‌ها (Q, QN)
                    port_name_upper == 'O' or  # نام‌های رایج خروجی
                    port_name_upper == 'Y' or  # (O, Y, Z, ZN)
                    port_name_upper == 'Z' or
                    port_name_upper == 'ZN'
            )
            # --- ✨✨✨ پایان اصلاح ✨✨✨ ---

            if is_output_port:
                G.add_node(pin_name, type='Pin_Output')
                G.add_edge(gate_name, pin_name)
                if wire not in wire_to_pins_map:
                    wire_to_pins_map[wire] = {'source_pin': None, 'sinks': []}
                wire_to_pins_map[wire]['source_pin'] = pin_name
            else:
                G.add_node(pin_name, type='Pin_Input')
                G.add_edge(pin_name, gate_name)
                if wire not in wire_to_pins_map:
                    wire_to_pins_map[wire] = {'source_pin': None, 'sinks': []}
                wire_to_pins_map[wire]['sinks'].append(pin_name)

    # 3. اتصال پین‌ها به یکدیگر بر اساس سیم‌ها
    for wire, pins in wire_to_pins_map.items():
        source_pin = pins['source_pin']
        if source_pin:
            for sink_pin in pins['sinks']:
                if G.has_node(source_pin) and G.has_node(sink_pin):
                    G.add_edge(source_pin, sink_pin)

    return G


# ##################################################################
# ########## TODO 2: پیاده‌سازی واقعی الگوریتم 1 #########
# ########## (بدون تغییر) ########################################
# ##################################################################
def _recursion(G: nx.DiGraph, current_cell: str, remaining_depth: int, max_depth: int, Direction: str) -> List[Dict]:
    """
    تابع کمکی بازگشتی برای generate_netlist_blocks (پیاده‌سازی RECURSION از Alg 1)
    """
    if remaining_depth <= 0: return []

    current_logic_level = (max_depth - remaining_depth) + 1
    found_nets = []

    if Direction == 'I':
        # جستجو در سمت ورودی (عقبگرد)
        try:
            input_pins = list(G.predecessors(current_cell))
            for vp in input_pins:  # vp = Current Input Pin
                source_pins = list(G.predecessors(vp))
                for vp_prime in source_pins:  # vp' = Next Output Pin
                    next_cells = list(G.predecessors(vp_prime))
                    for vc_prime in next_cells:  # vc' = Next Cell
                        # داده‌های نت مطابق با مقاله
                        net_data = [vc_prime, vp_prime, vp, current_cell, current_logic_level]
                        net_info = {
                            'net': net_data,
                            'children': _recursion(G, vc_prime, remaining_depth - 1, max_depth, 'I')
                        }
                        found_nets.append(net_info)
        except nx.NetworkXError:
            pass  # گره ترمینال

    elif Direction == 'O':
        # جستجو در سمت خروجی (پیشرو)
        try:
            output_pins = list(G.successors(current_cell))
            for vp in output_pins:  # vp = Current Output Pin
                sink_pins = list(G.successors(vp))
                for vp_prime in sink_pins:  # vp' = Next Input Pin
                    next_cells = list(G.successors(vp_prime))
                    for vc_prime in next_cells:  # vc' = Next Cell
                        # داده‌های نت مطابق با مقاله
                        net_data = [vc_prime, vp_prime, vp, current_cell, current_logic_level]
                        net_info = {
                            'net': net_data,
                            'children': _recursion(G, vc_prime, remaining_depth - 1, max_depth, 'O')
                        }
                        found_nets.append(net_info)
        except nx.NetworkXError:
            pass  # گره ترمینال

    return found_nets


def generate_netlist_blocks(pin_graph: nx.DiGraph, logic_level: int = 4) -> Dict:
    """
    پیاده‌سازی واقعی الگوریتم 1 مقاله.
    """
    all_blocks = {}
    cell_nodes = [n for n, d in pin_graph.nodes(data=True) if d.get('type') == 'Cell']

    for vc in cell_nodes:  # vc = گیت مرکزی
        block_tree = {
            'I': _recursion(pin_graph, vc, logic_level, logic_level, 'I'),
            'O': _recursion(pin_graph, vc, logic_level, logic_level, 'O')
        }
        all_blocks[vc] = block_tree

    return all_blocks


# ##################################################################
# ########## ✨ TODO 3: پیاده‌سازی واقعی الگوریتم 2 ✨ #########
# ##################################################################

def _find_all_root_to_leaf_paths(node_list: List[Dict]) -> List[List[List[Any]]]:
    """
    تابع کمکی DFS برای پیاده‌سازی بخش FLATTEN الگوریتم 2.
    تمام مسیرهای ریشه-به-برگ را در جنگل بلوک نت‌لیست پیدا می‌کند.
    """
    all_paths = []

    def dfs(node: Dict, current_path: List[List[Any]]):
        # 1. نت فعلی را به مسیر اضافه کن
        current_path.append(node['net'])

        # 2. اگر برگ بود (فرزندی نداشت)، مسیر کامل را ذخیره کن
        if not node['children']:
            all_paths.append(list(current_path))  # یک کپی از مسیر را ذخیره کن
        else:
            # 3. اگر برگ نبود، به جستجو در فرزندان ادامه بده
            for child in node['children']:
                dfs(child, current_path)

        # 4. بک‌ترک (Backtrack): نت فعلی را از مسیر حذف کن تا شاخه‌های دیگر بررسی شوند
        current_path.pop()

    # DFS را برای هر درخت در جنگل (لیست) شروع کن
    for root_node in node_list:
        dfs(root_node, [])

    return all_paths


def extract_pcp_traces(netlist_blocks: Dict) -> Dict[str, List[List[str]]]:
    """
    پیاده‌سازی واقعی الگوریتم 2 مقاله.
    بلوک‌های درختی را به ردیابی‌های PCP خطی (جملات) تبدیل می‌کند.
    """
    all_traces_map = {}

    # فرمت "کلمه" PCP مطابق با مقاله
    # ما از "___" به جای "_" برای جداسازی استفاده می‌کنیم تا از ابهام جلوگیری شود
    def create_pcp_word(v_in_p: str, v_c: str, v_out_p: str) -> str:
        return f"{v_in_p}___{v_c}___{v_out_p}"

    for center_gate, block in netlist_blocks.items():

        # 1. (FLATTEN) تمام مسیرهای ورودی و خروجی را پیدا کن
        input_paths = _find_all_root_to_leaf_paths(block['I'])
        output_paths = _find_all_root_to_leaf_paths(block['O'])

        # اگر هیچ مسیر ورودی یا خروجی وجود نداشت، ردیابی وجود ندارد
        if not input_paths or not output_paths:
            all_traces_map[center_gate] = []
            continue

        # 2. ترکیب هر مسیر ورودی با هر مسیر خروجی
        generated_traces_for_gate = []
        for path_I in input_paths:
            for path_O in output_paths:

                pcp_trace_words = []

                # --- الف: ساخت PCP های مسیر ورودی ---
                # (از لبه به سمت مرکز، Alg 2: lines 8-14)
                # مسیرها از قبل مرتب هستند (از مرکز به لبه)، پس معکوس می‌کنیم
                for i in range(len(path_I) - 1, 0, -1):
                    net_a = path_I[i - 1]  # نت نزدیکتر به مرکز (عمق x-1)
                    net_b = path_I[i]  # نت دورتر از مرکز (عمق x)

                    # PCP بر اساس Eq (1) و Fig 4b: [vp_b, vc_b, vp'_a]
                    # net_b[2] = vp_b (Current Input Pin)
                    # net_b[3] = vc_b (Current Cell)
                    # net_a[1] = vp'_a (Next Output Pin)
                    word = create_pcp_word(net_b[2], net_b[3], net_a[1])
                    pcp_trace_words.append(word)

                # --- ب: ساخت PCP مرکزی ---
                # (Alg 2: lines 15-18)
                net_a = path_I[0]  # نت ورودیِ متصل به مرکز
                net_b = path_O[0]  # نت خروجیِ متصل به مرکز

                # PCP: [net_a.v_p, net_a.v_c, net_b.v_p]
                # net_a[2] = vp_a (Current Input Pin)
                # net_a[3] = vc_a (Current Cell - The Center Gate)
                # net_b[2] = vp_b (Current Output Pin)
                center_word = create_pcp_word(net_a[2], net_a[3], net_b[2])
                pcp_trace_words.append(center_word)

                # --- ج: ساخت PCP های مسیر خروجی ---
                # (از مرکز به سمت لبه، Alg 2: lines 19-24)
                for i in range(len(path_O) - 1):
                    net_a = path_O[i]  # نت نزدیکتر به مرکز (عمق x-1)
                    net_b = path_O[i + 1]  # نت دورتر از مرکز (عمق x)

                    # PCP بر اساس Eq (1) و Fig 4b: [vp_a, vc_a, vp'_b]
                    # net_a[2] = vp_a (Current Output Pin)
                    # net_a[3] = vc_a (Current Cell)
                    # net_b[1] = vp'_b (Next Input Pin)
                    word = create_pcp_word(net_a[2], net_a[3], net_b[1])
                    pcp_trace_words.append(word)

                generated_traces_for_gate.append(pcp_trace_words)

        all_traces_map[center_gate] = generated_traces_for_gate

    return all_traces_map