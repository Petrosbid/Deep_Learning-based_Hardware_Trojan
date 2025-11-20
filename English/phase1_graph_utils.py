import networkx as nx
from typing import List, Dict, Set, Tuple, Any
from netlist_parser import Netlist

# Pin mapping for standardizing training data
PIN_NORMALIZATION = {
    'Q': 'OUT', 'QN': 'OUT', 'Q_N': 'OUT', 'Y': 'OUT', 'Z': 'OUT', 'O': 'OUT', 'SO': 'OUT',
    'A': 'IN', 'B': 'IN', 'C': 'IN', 'D': 'IN', 'E': 'IN',
    'DIN': 'IN', 'DIN1': 'IN', 'DIN2': 'IN', 'DIN3': 'IN', 'DIN4': 'IN',
    'SI': 'IN', 'D0': 'IN', 'D1': 'IN',
    'CLK': 'CLK', 'CK': 'CLK', 'RST': 'RST', 'S': 'SEL', 'SEL': 'SEL', 'EB': 'EN'
}


def get_normalized_pin(raw_pin: str) -> str:
    pin = raw_pin.split('___')[-1].upper()
    if pin in PIN_NORMALIZATION: return PIN_NORMALIZATION[pin]
    for key, val in PIN_NORMALIZATION.items():
        if pin.startswith(key): return val
    if pin.startswith(('I', 'A', 'D')): return 'IN'
    return 'OUT'


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
        # Use cleaned type (nand instead of nnd2s3)
        G.add_node(gate_name, type='Cell', cell_type=gate_obj.cell_type, is_trojan=gate_obj.is_trojan)

        for port in gate_obj.output_pins:
            wire = gate_obj.connections.get(port)
            if not wire: continue
            pin_name = f"{gate_name}___{port}"
            G.add_node(pin_name, type='Pin_Output')
            G.add_edge(gate_name, pin_name)
            if wire not in wire_to_pins_map: wire_to_pins_map[wire] = {'source_pin': None, 'sinks': []}
            wire_to_pins_map[wire]['source_pin'] = pin_name

        for port in gate_obj.input_pins:
            wire = gate_obj.connections.get(port)
            if not wire: continue
            pin_name = f"{gate_name}___{port}"
            G.add_node(pin_name, type='Pin_Input')
            G.add_edge(pin_name, gate_name)
            if wire not in wire_to_pins_map: wire_to_pins_map[wire] = {'source_pin': None, 'sinks': []}
            wire_to_pins_map[wire]['sinks'].append(pin_name)

    for wire, pins in wire_to_pins_map.items():
        source = pins['source_pin']
        if source:
            for sink in pins['sinks']:
                if G.has_node(source) and G.has_node(sink):
                    G.add_edge(source, sink)
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
                        # Store cell type in tuple for later use
                        cell_type = G.nodes[vc_prime].get('cell_type', 'unknown')
                        net_data = [vc_prime, vp_prime, vp, current_cell, current_logic_level, cell_type]
                        net_info = {'net': net_data,
                                    'children': _recursion(G, vc_prime, remaining_depth - 1, max_depth, 'I')}
                        found_nets.append(net_info)
        except:
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
                        cell_type = G.nodes[vc_prime].get('cell_type', 'unknown')
                        net_data = [vc_prime, vp_prime, vp, current_cell, current_logic_level, cell_type]
                        net_info = {'net': net_data,
                                    'children': _recursion(G, vc_prime, remaining_depth - 1, max_depth, 'O')}
                        found_nets.append(net_info)
        except:
            pass
    return found_nets


def generate_netlist_blocks(pin_graph: nx.DiGraph, logic_level: int = 4) -> Dict:
    all_blocks = {}
    cell_nodes = [n for n, d in pin_graph.nodes(data=True) if d.get('type') == 'Cell']
    for vc in cell_nodes:
        all_blocks[vc] = {
            'I': _recursion(pin_graph, vc, logic_level, logic_level, 'I'),
            'O': _recursion(pin_graph, vc, logic_level, logic_level, 'O'),
            'center_type': pin_graph.nodes[vc].get('cell_type', 'unknown')
        }
    return all_blocks


def _find_paths(node_list):
    all_paths = []

    def dfs(node, current_path):
        current_path.append(node['net'])
        if not node['children']:
            all_paths.append(list(current_path))
        else:
            for child in node['children']: dfs(child, current_path)
        current_path.pop()

    for root in node_list: dfs(root, [])
    if not all_paths and node_list:
        for root in node_list: all_paths.append([root['net']])
    return all_paths


def extract_pcp_traces(netlist_blocks: Dict) -> Dict[str, List[List[str]]]:
    all_traces_map = {}

    def create_word(pin1, c_type, pin2):
        return f"{get_normalized_pin(pin1)}___{c_type}___{get_normalized_pin(pin2)}"

    for center_gate, block in netlist_blocks.items():
        input_paths = _find_paths(block['I'])
        output_paths = _find_paths(block['O'])
        center_type = block['center_type']

        if not input_paths: input_paths = [[['DUMMY', 'IN', 'IN', center_gate, 1, 'dummy']]]
        if not output_paths: output_paths = [[['DUMMY', 'OUT', 'OUT', center_gate, 1, 'dummy']]]

        traces = []
        for pI in input_paths:
            for pO in output_paths:
                trace = []
                # Input side
                for i in range(len(pI) - 1, 0, -1):
                    net_a, net_b = pI[i - 1], pI[i]
                    # net_b[5] is cell_type
                    trace.append(create_word(net_b[2], net_b[5], net_a[1]))

                # Center
                trace.append(create_word(pI[0][2], center_type, pO[0][2]))

                # Output side
                for i in range(len(pO) - 1):
                    net_a, net_b = pO[i], pO[i + 1]
                    # net_a[5] is cell_type
                    trace.append(create_word(net_a[2], net_a[5], net_b[1]))

                traces.append(trace)
        all_traces_map[center_gate] = traces
    return all_traces_map