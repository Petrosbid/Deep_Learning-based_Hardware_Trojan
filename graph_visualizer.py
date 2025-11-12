import networkx as nx
import matplotlib.pyplot as plt
import netlist_parser  # (فاز ۰)
import phase1_graph_utils  # (فاز ۱)
from netlist_parser import Netlist


def visualize_subgraph(graph: nx.DiGraph, center_node: str, radius: int = 2):
    """
    یک زیرگراف کوچک در اطراف یک گیت مرکزی را رسم می‌کند.
    این بهترین راه برای دیباگ کردن است.
    """
    try:
        # یک زیرگراف شامل تمام همسایه‌ها تا عمق 'radius' ایجاد کن
        sub_graph = nx.ego_graph(graph, center_node, radius=radius, undirected=False)

        # استخراج برچسب‌ها و رنگ‌ها برای نمایش بهتر
        labels = {}
        colors = []
        for node in sub_graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data.get('type', 'Unknown')

            if node_type == 'Cell':
                labels[node] = f"CELL\n{node}\n({node_data.get('cell_type')})"
                colors.append('skyblue' if not node_data.get('is_trojan') else 'red')
            elif 'Pin' in node_type:
                # نام پین را کوتاه کن
                labels[node] = node.split('___')[-1]  # نمایش 'Q' به جای 'U1___Q'
                colors.append('lightgreen')
            elif 'Port' in node_type:
                labels[node] = f"PORT\n{node}"
                colors.append('gray')
            else:
                labels[node] = node
                colors.append('orange')

        print(f"Drawing subgraph around '{center_node}' with {sub_graph.number_of_nodes()} nodes...")

        plt.figure(figsize=(15, 10))
        # استفاده از 'spring_layout' برای چیدمان بهتر
        pos = nx.spring_layout(sub_graph, k=0.5, iterations=50)
        nx.draw(sub_graph, pos,
                with_labels=True,
                labels=labels,
                node_color=colors,
                node_size=2000,
                font_size=8,
                arrows=True,
                arrowstyle='->',
                arrowsize=12)
        plt.title(f"Pin-Graph Visualization (Radius {radius} around {center_node})")
        plt.show()

    except nx.NetworkXError as e:
        print(f"Error: Node '{center_node}' not found in graph. ({e})")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")


def main():
    # --- 1. یک فایل برای تست انتخاب کنید ---
    # (از فایلی که قبلاً ارسال کردید استفاده می‌کنیم)
    NETLIST_FILE = "./TRIT-TC/c2670_T001/c2670_T001.v"
    LOG_FILE = "./TRIT-TC/c2670_T001/log.txt"

    # گیت مرکزی برای تمرکز روی آن (یکی از گیت‌های تروجان)
    TARGET_GATE = "troj1_0U1"

    print("--- 1. Parsing ---")
    trojan_names = netlist_parser.parse_trojan_log(LOG_FILE)
    netlist = netlist_parser.parse_netlist(NETLIST_FILE, trojan_names)

    if not netlist:
        print("Failed to parse netlist.")
        return

    print("--- 2. Converting to Graph (Real) ---")
    # این همان تابعی است که می‌خواهیم تست کنیم
    pin_graph = phase1_graph_utils.convert_to_pin_graph(netlist)

    print(f"\nGraph created successfully!")
    print(f"Total Nodes: {pin_graph.number_of_nodes()}")
    print(f"Total Edges: {pin_graph.number_of_edges()}")

    # # --- 3. Visualizing Subgraph (روش پیشنهادی) ---
    visualize_subgraph(pin_graph, TARGET_GATE, radius=100)

    # --- (اختیاری) رسم کل گراف (هشدار: بسیار کند و نامرتب خواهد بود) ---
    # print("\nAttempting to draw full graph (may be slow/unreadable)...")
    # plt.figure(figsize=(20, 20))
    # nx.draw(pin_graph, node_size=10, font_size=6)
    # plt.title("Full Netlist Graph (Hairball)")
    # plt.show()


if __name__ == "__main__":
    main()