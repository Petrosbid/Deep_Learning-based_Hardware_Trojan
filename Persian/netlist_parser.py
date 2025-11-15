import re
from typing import List, Dict, Set, Optional


class Gate:
    """نگهداری اطلاعات یک گیت (یا assign)"""

    def __init__(self, instance_name, cell_type, is_trojan=False):
        self.instance_name = instance_name
        self.cell_type = cell_type
        self.is_trojan = is_trojan
        self.connections = {}  # {port: wire}

    def add_connection(self, port, wire):
        self.connections[port] = wire

    def __repr__(self):
        trojan_str = " (TROJAN)" if self.is_trojan else ""
        return (f"Gate(Name: '{self.instance_name}', "
                f"Type: '{self.cell_type}'{trojan_str})")


class Netlist:
    """نگهداری کل ساختار نت‌لیست (شبه-گراف مقاله)"""

    def __init__(self, module_name):
        self.module_name = module_name
        self.inputs = set()
        self.outputs = set()
        self.wires = set()
        self.gates = {}
    def add_gate(self, gate_obj):
        self.gates[gate_obj.instance_name] = gate_obj

    def get_gate_by_name(self, instance_name):
        return self.gates.get(instance_name)


def parse_trojan_log(log_file_path: str) -> Set[str]:
    """
    فایل log.txt را می‌خواند. اگر فایل وجود نداشته باشد،
    یک مجموعه خالی برمی‌گرداند (یعنی تروجانی نیست).
    """
    trojan_gate_names = set()
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        body_match = re.search(r'TROJAN BODY:\s*(.*)', content, re.DOTALL)
        if not body_match: return trojan_gate_names

        body = body_match.group(1)
        gate_regex = re.compile(r'^\s*[\w\d]+s\d+\s+([\w\d_]+)\s*\(')

        for line in body.splitlines():
            match = gate_regex.search(line)
            if match:
                instance_name = match.group(1)
                trojan_gate_names.add(instance_name)

    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"\n[Warning] Error parsing log {log_file_path}: {e}")

    return trojan_gate_names


def parse_netlist(netlist_file_path: str, trojan_gate_names: Set[str]) -> Optional[Netlist]:
    """
    فایل نت‌لیست .v را خط به خط می‌خواند و آبجکت Netlist می‌سازد.
    """
    re_module = re.compile(r'^\s*module\s+([\w\d_]+)', re.IGNORECASE)
    re_port = re.compile(r'^\s*(input|output)\s+(.+);', re.IGNORECASE)
    re_wire = re.compile(r'^\s*wire\s+(.+);', re.IGNORECASE)
    re_assign = re.compile(r'^\s*assign\s+([\w\d_\[\]:]+)\s*=\s*([\w\d_\[\]:\'b]+);', re.IGNORECASE)
    re_gate = re.compile(r'^\s*([\w\d_]+)\s+([\w\d_]+)\s*\((.*?)\);', re.DOTALL)
    re_port_conn = re.compile(r'\.([\w\d_]+)\s*\(([\w\d_\[\]:\'b]+)\)')

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

                match = re_wire.search(complete_line)
                if match:
                    wires_str = match.group(1)
                    wires = [w.strip() for w in wires_str.split(',') if w.strip()]
                    netlist_obj.wires.update(wires)
                    continue

                match = re_assign.search(complete_line)
                if match:
                    dest_wire, source_wire = match.groups()
                    instance_name = f"assign_{assign_counter}"
                    assign_counter += 1
                    gate_obj = Gate(instance_name, 'assign', instance_name in trojan_gate_names)
                    gate_obj.add_connection('Q', dest_wire)
                    gate_obj.add_connection('DIN', source_wire)
                    netlist_obj.add_gate(gate_obj)
                    continue

                match = re_gate.search(complete_line)
                if match:
                    cell_type, instance_name, connections_str = match.groups()
                    is_trojan = instance_name in trojan_gate_names
                    gate_obj = Gate(instance_name, cell_type, is_trojan)
                    for port_match in re_port_conn.finditer(connections_str):
                        gate_obj.add_connection(port_match.group(1), port_match.group(2))
                    netlist_obj.add_gate(gate_obj)
                    continue

        if netlist_obj and len(netlist_obj.gates) > 0:
            return netlist_obj
        else:
            return None
    except FileNotFoundError:
        print(f"\n[Error] Netlist file not found: {netlist_file_path}")
        return None
    except Exception as e:
        print(f"\n[Error] Failed to parse {netlist_file_path}: {e}")
        return None