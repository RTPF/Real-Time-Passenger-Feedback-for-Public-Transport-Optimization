import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "Grafos_Rutas_SVG")
if not os.path.exists(desktop_path):
    os.makedirs(desktop_path)

df = pd.read_csv(r"C:\Users\AlexisVM\Downloads\pasajeros_feedback.csv", encoding="latin-1")
df.dropna(inplace=True)

rutas_unicas = df["route_id"].unique()

sentiment_pos = {
    "Neutro": (0, 2),
    "Negativo": (-2, -2),
    "Positivo": (2, -2)
}

for ruta_objetivo in rutas_unicas:
    df_ruta = df[df["route_id"] == ruta_objetivo].copy()
    df_ruta["passenger_id_num"] = df_ruta["passenger_id"].astype("category").cat.codes

    G = nx.DiGraph()
    pos = {}

    G.add_node(ruta_objetivo)
    pos[ruta_objetivo] = (0, 0)

    for s, coord in sentiment_pos.items():
        nodo_s = f"{ruta_objetivo}_{s}"
        G.add_node(nodo_s)
        G.add_edge(ruta_objetivo, nodo_s)
        pos[nodo_s] = coord

    for _, row in df_ruta.iterrows():
        pid = f"P{row['passenger_id_num']}"
        sentimiento = row['sentiment']
        nodo_s = f"{ruta_objetivo}_{sentimiento}"
        G.add_node(pid)
        G.add_edge(nodo_s, pid)

    for sentimiento, centro in sentiment_pos.items():
        hijos = [n for n in G.successors(f"{ruta_objetivo}_{sentimiento}") if n.startswith("P")]
        n = len(hijos)
        if n == 0:
            continue
        angulos = np.linspace(-np.pi/4, np.pi/4, n)
        for i, h in enumerate(hijos):
            x = centro[0] + 0.8 * np.cos(angulos[i])
            y = centro[1] + 0.8 * np.sin(angulos[i])
            pos[h] = (x, y)

    node_colors = []
    for n in G.nodes():
        if n == ruta_objetivo:
            node_colors.append("orange")
        elif "Neutro" in n:
            node_colors.append("gray")
        elif "Negativo" in n:
            node_colors.append("red")
        elif "Positivo" in n:
            node_colors.append("green")
        else:
            node_colors.append("lightblue")

    plt.figure(figsize=(8, 8), facecolor="white")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors="black")
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=1)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", verticalalignment="top")

    plt.title(f"Grafo de la Ruta {ruta_objetivo}")
    plt.axis("off")

    plt.gca().set_facecolor("white")

    svg_path = os.path.join(desktop_path, f"grafo_ruta_{ruta_objetivo}.svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()

print(f"Los grafos se han guardado como imágenes SVG en la carpeta: {desktop_path}")