import re
from typing import List, Dict, Set, Optional

# تنظیمات تمیزکاری نام گیت‌ها
IGNORED_SUFFIXES = [r's\d+$', r'd\d+$', r'x\d+$', r'_[\d\w]+$', r'\d+$']
TYPE_MAPPING = {
    'nnd': 'nand', 'inv': 'not', 'buf': 'buffer',
    'dff': 'dff', 'sdff': 'dff', 'xnr': 'xnor',
    'hi1': 'tiehi', 'lo1': 'tielo', 'assign': 'buffer'
}

# تنظیمات تشخیص جهت پین‌ها
KNOWN_OUTPUTS = {'Y', 'Q', 'QN', 'Q_N', 'Z', 'ZN', 'O', 'OUT', 'SO', 'CO', 'S', 'SUM', 'RESULT', 'R'}
KNOWN_INPUTS_PREFIXES = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'IN', 'SEL', 'CLK', 'CK', 'RST', 'EN', 'TE',
                         'ADDR', 'DATA', 'SIN', 'SCAN')


class Gate:
    def __init__(self, instance_name, cell_type, is_trojan=False):
        self.instance_name = instance_name
        self.original_type = cell_type
        self.is_trojan = is_trojan
        # تمیز کردن نام گیت
        self.cell_type = self._clean_type(cell_type)
        self.connections = {}  # {port: wire}
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
        """تشخیص هوشمند جهت پین‌ها برای ساخت گراف"""
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
            elif temp_inputs:  # Fallback
                temp_outputs.append(temp_inputs.pop())
        else:
            temp_inputs.extend(unknowns)

        self.output_pins = temp_outputs
        self.input_pins = temp_inputs

    def add_connection(self, port, wire):
        self.connections[port] = wire


class Netlist:
    def __init__(self, module_name):
        self.module_name = module_name
        self.inputs = set()
        self.outputs = set()
        self.wires = set()
        self.gates = {}  # Map: instance_name -> Gate Obj

    def add_gate(self, gate_obj):
        self.gates[gate_obj.instance_name] = gate_obj


def parse_trojan_log(log_file_path: str) -> Set[str]:
    trojan_gate_names = set()
    if not log_file_path: return trojan_gate_names
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        body_match = re.search(r'TROJAN BODY:\s*(.*)', content, re.DOTALL)
        if not body_match: return trojan_gate_names
        body = body_match.group(1)
        gate_regex = re.compile(r'^\s*[\w\d]+s\d+\s+([\w\d_]+)\s*\(')  # تطبیق با فرمت لاگ
        for line in body.splitlines():
            match = gate_regex.search(line)
            if match: trojan_gate_names.add(match.group(1))
    except:
        pass
    return trojan_gate_names


def parse_netlist(netlist_file_path: str, trojan_gate_names: Set[str]) -> Optional[Netlist]:
    re_module = re.compile(r'^\s*module\s+([\w\d_]+)', re.IGNORECASE)
    re_port = re.compile(r'^\s*(input|output)\s+(.+);', re.IGNORECASE)
    re_wire = re.compile(r'^\s*wire\s+(.+);', re.IGNORECASE)
    re_assign = re.compile(r'^\s*assign\s+([\w\d_\[\]:]+)\s*=\s*([\w\d_\[\]:\'b]+);', re.IGNORECASE)
    re_gate = re.compile(r'^\s*([\w\d_]+)\s+([\w\d_:\\]+)\s*\((.*?)\);', re.DOTALL)
    re_port_conn = re.compile(r'\.([\w\d_]+)\s*\(([\w\d_\[\]:\'b]*)\)')

    netlist_obj = None
    assign_counter = 0

    try:
        with open(netlist_file_path, 'r', encoding='utf-8') as f:
            # خواندن کل فایل و حذف کامنت‌ها برای پارس بهتر
            content = f.read()
            content = re.sub(r'//.*', '', content)
            statements = content.split(';')

            for stmt in statements:
                stmt = stmt.strip()
                if not stmt: continue

                if netlist_obj is None:
                    match = re_module.search(stmt)
                    if match: netlist_obj = Netlist(module_name=match.group(1))
                    continue

                match = re_port.search(stmt + ';')
                if match:
                    port_type, ports_str = match.groups()
                    ports = [p.strip() for p in ports_str.split(',') if p.strip()]
                    if port_type.lower() == 'input':
                        netlist_obj.inputs.update(ports)
                    else:
                        netlist_obj.outputs.update(ports)
                    continue

                match = re_wire.search(stmt + ';')
                if match:
                    wires_str = match.group(1)
                    wires = [w.strip() for w in wires_str.split(',') if w.strip()]
                    netlist_obj.wires.update(wires)
                    continue

                match = re_assign.search(stmt + ';')
                if match:
                    dest_wire, source_wire = match.groups()
                    instance_name = f"assign_{assign_counter}"
                    assign_counter += 1
                    gate_obj = Gate(instance_name, 'buffer', instance_name in trojan_gate_names)
                    gate_obj.add_connection('Q', dest_wire)
                    gate_obj.add_connection('A', source_wire)
                    gate_obj.infer_pin_directions()
                    netlist_obj.add_gate(gate_obj)
                    continue

                match = re_gate.search(stmt + ';')
                if match:
                    cell_type, instance_name, connections_str = match.groups()
                    instance_name = instance_name.replace('\\', '').strip()
                    is_trojan = instance_name in trojan_gate_names

                    gate_obj = Gate(instance_name, cell_type, is_trojan)
                    for port_match in re_port_conn.finditer(connections_str):
                        gate_obj.add_connection(port_match.group(1), port_match.group(2))

                    gate_obj.infer_pin_directions()
                    netlist_obj.add_gate(gate_obj)
                    continue

        if netlist_obj and len(netlist_obj.gates) > 0: return netlist_obj
        return None
    except Exception as e:
        print(f"Error parsing {netlist_file_path}: {e}")
        return None