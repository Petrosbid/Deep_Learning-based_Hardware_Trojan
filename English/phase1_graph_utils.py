#
# phase1_graph_utils.py
# (Phase 1: Convert to Pin-to-Pin Graph and Feature Extraction)
#
import networkx as nx
from typing import List, Dict, Set, Tuple, Any
from netlist_parser import Netlist, Gate  # Import from previous file


# ##################################################################
# ########## TODO 1: Real Cell-Pin Splitter Implementation #########
# ########## (Unchanged) ########################################
# ##################################################################
def convert_to_pin_graph(netlist: Netlist) -> nx.DiGraph:
    """
    Converts netlist to a real pin-to-pin graph (according to Figure 3c).
    (Improved version with smarter pin detection assumption)
    """
    G = nx.DiGraph()
    wire_to_pins_map = {}

    # 1. Add Module Input/Output Ports
    for port in netlist.inputs:
        G.add_node(port, type='Port_Input')
        wire_to_pins_map[port] = {'source_pin': port, 'sinks': []}
    for port in netlist.outputs:
        G.add_node(port, type='Port_Output')
        if port not in wire_to_pins_map:
            wire_to_pins_map[port] = {'source_pin': None, 'sinks': []}
        wire_to_pins_map[port]['sinks'].append(port)

    # 2. Add Cell and Pin Nodes and Internal Edges
    for gate_name, gate_obj in netlist.gates.items():
        # is_trojan=False because we are in detection mode
        G.add_node(gate_name, type='Cell', cell_type=gate_obj.cell_type, is_trojan=False)

        for port, wire in gate_obj.connections.items():
            pin_name = f"{gate_name}___{port}"

            # --- ✨✨✨ Improved code ✨✨✨ ---
            # Smarter pin output detection assumption
            port_name_upper = port.upper()
            is_output_port = (
                    port_name_upper.startswith('Q') or  # For flip-flops (Q, QN)
                    port_name_upper == 'O' or  # Common output names
                    port_name_upper == 'Y' or  # (O, Y, Z, ZN)
                    port_name_upper == 'Z' or
                    port_name_upper == 'ZN'
            )
            # --- ✨✨✨ End of improvement ✨✨✨ ---

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

    # 3. Connect Pins to Each Other Based on Wires
    for wire, pins in wire_to_pins_map.items():
        source_pin = pins['source_pin']
        if source_pin:
            for sink_pin in pins['sinks']:
                if G.has_node(source_pin) and G.has_node(sink_pin):
                    G.add_edge(source_pin, sink_pin)

    return G


# ##################################################################
# ########## TODO 2: Real Algorithm 1 Implementation #########
# ########## (Unchanged) ########################################
# ##################################################################
def _recursion(G: nx.DiGraph, current_cell: str, remaining_depth: int, max_depth: int, Direction: str) -> List[Dict]:
    """
    Recursive helper function for generate_netlist_blocks (RECURSION implementation from Alg 1)
    """
    if remaining_depth <= 0: return []

    current_logic_level = (max_depth - remaining_depth) + 1
    found_nets = []

    if Direction == 'I':
        # Search in input side (backward)
        try:
            input_pins = list(G.predecessors(current_cell))
            for vp in input_pins:  # vp = Current Input Pin
                source_pins = list(G.predecessors(vp))
                for vp_prime in source_pins:  # vp' = Next Output Pin
                    next_cells = list(G.predecessors(vp_prime))
                    for vc_prime in next_cells:  # vc' = Next Cell
                        # Net data as per paper
                        net_data = [vc_prime, vp_prime, vp, current_cell, current_logic_level]
                        net_info = {
                            'net': net_data,
                            'children': _recursion(G, vc_prime, remaining_depth - 1, max_depth, 'I')
                        }
                        found_nets.append(net_info)
        except nx.NetworkXError:
            pass  # Terminal node

    elif Direction == 'O':
        # Search in output side (forward)
        try:
            output_pins = list(G.successors(current_cell))
            for vp in output_pins:  # vp = Current Output Pin
                sink_pins = list(G.successors(vp))
                for vp_prime in sink_pins:  # vp' = Next Input Pin
                    next_cells = list(G.successors(vp_prime))
                    for vc_prime in next_cells:  # vc' = Next Cell
                        # Net data as per paper
                        net_data = [vc_prime, vp_prime, vp, current_cell, current_logic_level]
                        net_info = {
                            'net': net_data,
                            'children': _recursion(G, vc_prime, remaining_depth - 1, max_depth, 'O')
                        }
                        found_nets.append(net_info)
        except nx.NetworkXError:
            pass  # Terminal node

    return found_nets


def generate_netlist_blocks(pin_graph: nx.DiGraph, logic_level: int = 4) -> Dict:
    """
    Real implementation of paper's Algorithm 1.
    """
    all_blocks = {}
    cell_nodes = [n for n, d in pin_graph.nodes(data=True) if d.get('type') == 'Cell']

    for vc in cell_nodes:  # vc = Center gate
        block_tree = {
            'I': _recursion(pin_graph, vc, logic_level, logic_level, 'I'),
            'O': _recursion(pin_graph, vc, logic_level, logic_level, 'O')
        }
        all_blocks[vc] = block_tree

    return all_blocks


# ##################################################################
# ########## ✨ TODO 3: Real Algorithm 2 Implementation ✨ #########
# ##################################################################

def _find_all_root_to_leaf_paths(node_list: List[Dict]) -> List[List[List[Any]]]:
    """
    DFS helper function for implementing FLATTEN part of Algorithm 2.
    Finds all root-to-leaf paths in the netlist block forest.
    """
    all_paths = []

    def dfs(node: Dict, current_path: List[List[Any]]):
        # 1. Add current net to path
        current_path.append(node['net'])

        # 2. If leaf (no children), save complete path
        if not node['children']:
            all_paths.append(list(current_path))  # Save a copy of the path
        else:
            # 3. If not leaf, continue searching in children
            for child in node['children']:
                dfs(child, current_path)

        # 4. Backtrack: Remove current net from path so other branches can be explored
        current_path.pop()

    # Start DFS for each tree in the forest (list)
    for root_node in node_list:
        dfs(root_node, [])

    return all_paths


def extract_pcp_traces(netlist_blocks: Dict) -> Dict[str, List[List[str]]]:
    """
    Real implementation of paper's Algorithm 2.
    Converts tree blocks to linear PCP traces (sentences).
    """
    all_traces_map = []

    # PCP "word" format as per paper
    # We use "___" instead of "_" to avoid ambiguity
    def create_pcp_word(v_in_p: str, v_c: str, v_out_p: str) -> str:
        return f"{v_in_p}___{v_c}___{v_out_p}"

    for center_gate, block in netlist_blocks.items():

        # 1. (FLATTEN) Find all input and output paths
        input_paths = _find_all_root_to_leaf_paths(block['I'])
        output_paths = _find_all_root_to_leaf_paths(block['O'])

        # If no input or output paths exist, no trace exists
        if not input_paths or not output_paths:
            all_traces_map[center_gate] = []
            continue

        # 2. Combine each input path with each output path
        generated_traces_for_gate = []
        for path_I in input_paths:
            for path_O in output_paths:

                pcp_trace_words = []

                # --- a: Create Input Path PCPs ---
                # (from edge towards center, Alg 2: lines 8-14)
                # Paths are already sorted (from center to edge), so we reverse
                for i in range(len(path_I) - 1, 0, -1):
                    net_a = path_I[i - 1]  # Net closer to center (depth x-1)
                    net_b = path_I[i]  # Net farther from center (depth x)

                    # PCP based on Eq (1) and Fig 4b: [vp_b, vc_b, vp'_a]
                    # net_b[2] = vp_b (Current Input Pin)
                    # net_b[3] = vc_b (Current Cell)
                    # net_a[1] = vp'_a (Next Output Pin)
                    word = create_pcp_word(net_b[2], net_b[3], net_a[1])
                    pcp_trace_words.append(word)

                # --- b: Create Center PCP ---
                # (Alg 2: lines 15-18)
                net_a = path_I[0]  # Input net connected to center
                net_b = path_O[0]  # Output net connected to center

                # PCP: [net_a.v_p, net_a.v_c, net_b.v_p]
                # net_a[2] = vp_a (Current Input Pin)
                # net_a[3] = vc_a (Current Cell - The Center Gate)
                # net_b[2] = vp_b (Current Output Pin)
                center_word = create_pcp_word(net_a[2], net_a[3], net_b[2])
                pcp_trace_words.append(center_word)

                # --- c: Create Output Path PCPs ---
                # (from center towards edge, Alg 2: lines 19-24)
                for i in range(len(path_O) - 1):
                    net_a = path_O[i]  # Net closer to center (depth x-1)
                    net_b = path_O[i + 1]  # Net farther from center (depth x)

                    # PCP based on Eq (1) and Fig 4b: [vp_a, vc_a, vp'_b]
                    # net_a[2] = vp_a (Current Output Pin)
                    # net_a[3] = vc_a (Current Cell)
                    # net_b[1] = vp'_b (Next Input Pin)
                    word = create_pcp_word(net_a[2], net_a[3], net_b[1])
                    pcp_trace_words.append(word)

                generated_traces_for_gate.append(pcp_trace_words)

        all_traces_map[center_gate] = generated_traces_for_gate

    return all_traces_map