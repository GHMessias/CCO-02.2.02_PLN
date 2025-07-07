import networkx as nx
import matplotlib.pyplot as plt
import os

import os
import networkx as nx
import matplotlib.pyplot as plt

def evaluate_graph(edge_index, num_nodes, classes, plot_path='degree_distribution.png'):
    """
    Avalia múltiplas métricas de um grafo a partir do edge_index.

    Parâmetros:
    -----------
    edge_index : torch.Tensor
        Tensor [2, num_edges] com as conexões do grafo.

    num_nodes : int
        Número de nós do grafo.

    classes : list or array-like
        Lista com as classes de cada nó (usado na assortatividade).

    plot_path : str
        Caminho do arquivo para salvar o gráfico de distribuição de grau.

    Retorna:
    --------
    dict com as métricas:
        - 'assortativity'
        - 'density'
        - 'modularity'
        - 'num_connected_components'
        - 'num_edges'
        - 'degree_plot_path'
    """
    # Constrói o grafo
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    G.add_edges_from(edges)

    for idx, c in enumerate(classes):
        G.nodes[idx]['classe'] = c

    # Assortatividade por classe
    assortativity = nx.degree_assortativity_coefficient(G, 'classe')

    # Densidade
    density = nx.density(G)

    # Modularidade (requer python-louvain)
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(partition, G)
    except ImportError:
        modularity = None  # ou lance um erro, se preferir

    # Número de componentes conexas
    num_components = nx.number_connected_components(G)

    # Número de arestas
    num_edges = G.number_of_edges()

    # Plot da distribuição de grau
    degrees = [degree for _, degree in G.degree()]
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=range(1, max(degrees)+2), align='left', edgecolor='black')
    plt.xlabel("Grau do nó")
    plt.ylabel("Número de nós")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        'assortativity': assortativity,
        'density': density,
        'modularity': modularity,
        'num_connected_components': num_components,
        'num_edges': num_edges,
        'degree_plot_path': os.path.abspath(plot_path)
    }
