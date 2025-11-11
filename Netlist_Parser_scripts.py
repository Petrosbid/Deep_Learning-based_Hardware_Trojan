import re
import pprint  # برای چاپ خوانا


class Gate:
    """
    کلاسی برای نگهداری اطلاعات یک گیت (یا یک assign)
    """

    def __init__(self, instance_name, cell_type, is_trojan=False):
        self.instance_name = instance_name  # نام نمونه (مانند U1 یا trojan1_0)
        self.cell_type = cell_type  # نوع گیت (مانند nor2s1 یا assign)
        self.is_trojan = is_trojan  # آیا بخشی از تروجان است؟
        # دیکشنری پورت‌ها به سیم‌ها
        # مثال: {'Q': 'N3038', 'DIN1': 'n279', 'DIN2': 'n280'}
        self.connections = {}

    def add_connection(self, port, wire):
        self.connections[port] = wire

    def __repr__(self):
        # بازنمایی رشته‌ای برای چاپ خوانا
        trojan_str = " (TROJAN)" if self.is_trojan else ""
        return (f"Gate(Name: '{self.instance_name}', "
                f"Type: '{self.cell_type}'{trojan_str}\n"
                f"\tConnections: {self.connections})")

class Netlist:
    """
    کلاس اصلی که کل نت‌لیست را نگهداری می‌کند.
    این کلاس همان "Netlist Pseudo-Graph" در مقاله است.
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self.inputs = set()
        self.outputs = set()
        self.wires = set()
        # دیکشنری برای دسترسی سریع به گیت‌ها با نام نمونه
        self.gates = {}  # {instance_name: Gate_Object}

    def add_gate(self, gate_obj):
        self.gates[gate_obj.instance_name] = gate_obj

    def get_gate_by_name(self, instance_name):
        return self.gates.get(instance_name)

    def __repr__(self):
        return (f"Netlist(Module: '{self.module_name}'\n"
                f"  Inputs: {len(self.inputs)}, "
                f"Outputs: {len(self.outputs)}, "
                f"Wires: {len(self.wires)}, "
                f"Gates: {len(self.gates)})")



# --- 2. پارسر فایل Log تروجان ---

def parse_trojan_log(log_file_path):
    """
    فایل log.txt را می‌خواند تا نام گیت‌های تروجان را استخراج کند.

    """
    trojan_gate_names = set()
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()

        # پیدا کردن بخش TROJAN BODY
        body_match = re.search(r'TROJAN BODY:\s*(.*)', content, re.DOTALL)
        if not body_match:
            print(f"Warning: 'TROJAN BODY:' section not found in {log_file_path}")
            return trojan_gate_names

        body = body_match.group(1)

        # استخراج نام نمونه (instance name) از هر خط
        # مثال: '  nor3s1 troj1_0U1 ( ...' -> 'troj1_0U1'
        # [cite: 361, 362, 363, 364, 365]
        gate_regex = re.compile(r'^\s*[\w\d]+s\d+\s+([\w\d_]+)\s*\(')

        for line in body.splitlines():
            match = gate_regex.search(line)
            if match:
                instance_name = match.group(1)
                trojan_gate_names.add(instance_name)

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
    except Exception as e:
        print(f"Error parsing log file: {e}")

    return trojan_gate_names


# --- 3. پارسر اصلی نت‌لیست ---

def parse_netlist(netlist_file_path, trojan_gate_names):
    """
    فایل نت‌لیست .v را خط به خط می‌خواند و اشیاء Netlist و Gate را می‌سازد.
    """

    # --- تعریف Regex‌های مورد نیاز ---
    # Regex برای 'module c2670 ( ... );' [cite: 1]
    re_module = re.compile(r'^\s*module\s+([\w\d_]+)', re.IGNORECASE)

    # Regex برای 'input N1, N2, ...;' [cite: 7]
    re_port = re.compile(r'^\s*(input|output)\s+(.+);', re.IGNORECASE)

    # Regex برای 'wire n279, ...;' [cite: 13, 24]
    re_wire = re.compile(r'^\s*wire\s+(.+);', re.IGNORECASE)

    # Regex برای 'assign N398 = N219;' [cite: 24]
    re_assign = re.compile(r'^\s*assign\s+([\w\d_\[\]:]+)\s*=\s*([\w\d_\[\]:\'b]+);', re.IGNORECASE)

    # Regex برای گیت: 'nor2s1 U1 ( .Q(N3038), .DIN1(n279), .DIN2(n280) );' [cite: 43]
    re_gate = re.compile(r'^\s*([\w\d_]+)\s+([\w\d_]+)\s*\((.*?)\);', re.DOTALL)

    # Regex برای پورت‌ها: '.Q(N3038)'
    re_port_conn = re.compile(r'\.([\w\d_]+)\s*\(([\w\d_\[\]:\'b]+)\)')

    netlist_obj = None
    buffer = ""  # بافری برای نگهداری خطوط ناقص (مانند تعریف پورت‌ها)
    assign_counter = 0  # برای نام‌گذاری یکتا به assign ها

    try:
        with open(netlist_file_path, 'r') as f:
            for line in f:
                line = line.strip()

                # نادیده گرفتن خطوط خالی و کامنت‌ها
                if not line or line.startswith('//'):
                    continue

                # مدیریت تعریف‌های چندخطی (مانند لیست پورت‌ها)
                buffer += " " + line

                # اگر خط به ; ختم نشود، یعنی ادامه دارد
                if not buffer.endswith(';'):
                    continue

                # --- پردازش خط کامل شده ---
                complete_line = buffer.strip()
                buffer = ""  # بافر را خالی کن

                # 1. پیدا کردن Module
                match = re_module.search(complete_line)
                if match:
                    netlist_obj = Netlist(module_name=match.group(1))
                    continue  # برو خط بعد

                if netlist_obj is None:
                    continue  # تا زمانی که module تعریف نشده، ادامه نده

                # 2. پیدا کردن Input/Output
                match = re_port.search(complete_line)
                if match:
                    port_type = match.group(1).lower()
                    ports_str = match.group(2)
                    # تمام پورت‌های لیست شده با کاما را جدا کن
                    ports = [p.strip() for p in ports_str.split(',') if p.strip()]
                    if port_type == 'input':
                        netlist_obj.inputs.update(ports)
                    elif port_type == 'output':
                        netlist_obj.outputs.update(ports)
                    continue

                # 3. پیدا کردن Wire
                match = re_wire.search(complete_line)
                if match:
                    wires_str = match.group(1)
                    wires = [w.strip() for w in wires_str.split(',') if w.strip()]
                    netlist_obj.wires.update(wires)
                    continue

                # 4. پیدا کردن Assign
                match = re_assign.search(complete_line)
                if match:
                    dest_wire = match.group(1)
                    source_wire = match.group(2)
                    instance_name = f"assign_{assign_counter}"
                    assign_counter += 1

                    gate_obj = Gate(instance_name, cell_type='assign', is_trojan=(instance_name in trojan_gate_names))
                    gate_obj.add_connection('Q', dest_wire)  # خروجی
                    gate_obj.add_connection('DIN', source_wire)  # ورودی
                    netlist_obj.add_gate(gate_obj)
                    continue

                # 5. پیدا کردن Gate Instantiation
                match = re_gate.search(complete_line)
                if match:
                    cell_type = match.group(1)
                    instance_name = match.group(2)
                    connections_str = match.group(3)

                    # بررسی اینکه آیا این گیت تروجان است یا نه
                    is_trojan = instance_name in trojan_gate_names

                    gate_obj = Gate(instance_name, cell_type, is_trojan)

                    # استخراج تمام اتصالات پورت‌ها
                    for port_match in re_port_conn.finditer(connections_str):
                        port_name = port_match.group(1)
                        wire_name = port_match.group(2)
                        gate_obj.add_connection(port_name, wire_name)

                    netlist_obj.add_gate(gate_obj)
                    continue

        return netlist_obj

    except FileNotFoundError:
        print(f"Error: Netlist file not found at {netlist_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return None


# --- 4. بخش اجرایی اصلی ---

if __name__ == "__main__":
    NETLIST_FILE = "c2670_T001.v"
    LOG_FILE = "log.txt"

    print(f"--- 1. Parsing Trojan Log File: {LOG_FILE} ---")
    trojan_names = parse_trojan_log(LOG_FILE)
    print(f"Found {len(trojan_names)} Trojan gates:")
    print(trojan_names)
    print("-" * 40)

    print(f"--- 2. Parsing Netlist File: {NETLIST_FILE} ---")
    parsed_netlist = parse_netlist(NETLIST_FILE, trojan_names)

    if parsed_netlist:
        print("Netlist parsed successfully.")
        print(parsed_netlist)
        print("-" * 40)

        # --- 3. نمایش نمونه‌ای از نتایج ---
        print("--- 3. Sample Parsed Gates ---")

        # نمایش یک گیت عادی (بر اساس فایل نمونه)
        gate_u1 = parsed_netlist.get_gate_by_name("U1")
        if gate_u1:
            print("\nExample: Normal Gate (U1)")
            pprint.pprint(gate_u1)

        # نمایش یک گیت تروجان (بر اساس فایل نمونه)
        gate_trojan = parsed_netlist.get_gate_by_name("trojan1_0")
        if gate_trojan:
            print("\nExample: Trojan Gate (trojan1_0)")
            pprint.pprint(gate_trojan)

        # نمایش یک گیت تروجان دیگر
        gate_troj_u2 = parsed_netlist.get_gate_by_name("troj1_0U2")
        if gate_troj_u2:
            print("\nExample: Trojan Gate (troj1_0U2)")
            pprint.pprint(gate_troj_u2)

        print("\n--- Parser is ready for the next step (Cell-Pin Splitter) ---")