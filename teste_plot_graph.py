import pandas as pd
import torch
import networkx as nx

def edge_index_to_networkx(edge_index: torch.Tensor, classes=None, directed=False):
    """
    Converte um edge_index (PyTorch Tensor) em um grafo NetworkX, incluindo a classe de cada nó.

    Parâmetros:
    ------------
    edge_index : torch.Tensor
        Tensor de shape [2, num_edges], onde cada coluna representa uma aresta (origem -> destino).

    classes : list, array ou torch.Tensor (opcional)
        Lista ou vetor com a classe de cada nó (de tamanho igual ao número de nós).
        O valor será adicionado como atributo 'class' em cada nó.

    directed : bool
        Se True, cria um grafo direcionado. Caso contrário, o grafo será não direcionado.

    Retorna:
    ---------
    G : networkx.Graph ou networkx.DiGraph
        Grafo construído a partir do edge_index, com classes adicionadas aos nós.
    """
    if edge_index.size(0) != 2:
        raise ValueError("edge_index deve ter shape [2, num_edges]")

    # Cria o grafo
    G = nx.DiGraph() if directed else nx.Graph()

    # Adiciona arestas
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    # Inferência do número de nós
    num_nodes = max(edge_index.max().item() + 1, G.number_of_nodes())

    # Adiciona nós com classe (se fornecido)
    for i in range(num_nodes):
        class_label = classes[i] if classes is not None else None
        G.add_node(i, class_=class_label)

    return G

import matplotlib.pyplot as plt
import networkx as nx
import os

def draw_all_graph_layouts(G, output_dir='outputs', node_size=300, font_size=10, dpi=300):
    """
    Gera e salva visualizações do grafo em diferentes layouts, com nós coloridos por classe.

    Parâmetros:
    ------------
    G : networkx.Graph
        Grafo com atributo 'class_' em cada nó.

    output_dir : str
        Diretório onde os arquivos PNG serão salvos.

    node_size : int
        Tamanho dos nós no gráfico.

    font_size : int
        Tamanho das labels dos nós.

    dpi : int
        Resolução dos arquivos salvos.

    Retorna:
    --------
    None (salva os arquivos como imagens)
    """
    layouts = {
        'spring': nx.spring_layout(G, seed=42),
        'kamada_kawai': nx.kamada_kawai_layout(G),
        'circular': nx.circular_layout(G),
        'spectral': nx.spectral_layout(G),
        'shell': nx.shell_layout(G),
        'random': nx.random_layout(G),
    }

    # Pega classes dos nós
    class_labels = [data.get('class_', 'unknown') for _, data in G.nodes(data=True)]

    # Mapeia classes para cores
    unique_classes = sorted(set(class_labels))
    class_to_color = {cls: i for i, cls in enumerate(unique_classes)}
    node_colors = [class_to_color[cls] for cls in class_labels]
    cmap = plt.cm.get_cmap("Set1", len(unique_classes))

    # Cria diretório se necessário
    os.makedirs(output_dir, exist_ok=True)

    for layout_name, pos in layouts.items():
        plt.figure(figsize=(10, 7))
        nx.draw(
            G, pos,
            with_labels=False,
            node_color=node_colors,
            cmap=cmap,
            node_size=node_size,
            font_size=font_size,
            edge_color='gray'
        )

        # Legenda
        for cls, color_id in class_to_color.items():
            plt.scatter([], [], color=cmap(color_id), label=str(cls))
        plt.legend(title='Classe', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title(f"Grafo - Layout: {layout_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/output_{layout_name}.png", dpi=dpi)
        plt.close()

    print(f"Arquivos salvos em: {os.path.abspath(output_dir)}")

import matplotlib.pyplot as plt
import networkx as nx

def draw_graph_with_classes(G, classes, layout='spring', figsize=(10, 7), node_size=300, font_size=10):
    """
    Desenha um grafo NetworkX com os nós coloridos por classe baseada em vetor externo.

    Parâmetros:
    ------------
    G : networkx.Graph
        Grafo (a estrutura pode ou não ter atributo de classe nos nós).

    classes : list, array ou torch.Tensor
        Lista com a classe (-1 ou 1) de cada vértice. Deve ter len igual ao número de nós em G.

    layout : str
        Tipo de layout: 'spring', 'kamada_kawai', 'circular', 'spectral', 'shell', 'random'.

    figsize : tuple
        Tamanho da figura em polegadas (largura, altura).

    node_size : int
        Tamanho dos nós no gráfico.

    font_size : int
        Tamanho das labels dos nós.

    Retorna:
    --------
    None (exibe o gráfico na tela)
    """
    # Validação
    if len(classes) != G.number_of_nodes():
        raise ValueError("Tamanho do vetor 'classes' deve ser igual ao número de nós do grafo.")

    # Layouts disponíveis
    layout_funcs = {
        'spring': nx.spring_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'circular': nx.circular_layout,
        'spectral': nx.spectral_layout,
        'shell': nx.shell_layout,
        'random': nx.random_layout
    }

    if layout not in layout_funcs:
        raise ValueError(f"Layout '{layout}' não reconhecido.")

    pos = layout_funcs[layout](G, seed=42) if layout == 'spring' else layout_funcs[layout](G)

    # Mapeia classe para cor
    node_colors = []
    for cls in classes:
        if cls == -1:
            node_colors.append('blue')
        elif cls == 1:
            node_colors.append('red')
        else:
            node_colors.append('gray')

    # Desenha
    plt.figure(figsize=figsize)
    nx.draw(
        G, pos,
        with_labels=False,
        node_color=node_colors,
        node_size=node_size,
        edge_color='gray'
    )

    # Legenda
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Classe -1', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Classe 1', markerfacecolor='red', markersize=10)
    ]
    plt.legend(handles=handles, title='Classe', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f"Grafo com layout '{layout}'")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# y = pd.read_csv('dataset/Fact_checked_news.tsv', sep = '\t')['label']

data = torch.load('logs/2025-07-06_15-19-23/graphs/yake.pt', weights_only=False)

g = edge_index_to_networkx(data.edge_index)

print(g)

# draw_all_graph_layouts(g)

draw_graph_with_classes(g, classes = data.y.detach().numpy(), node_size=20)
draw_graph_with_classes(g, classes = data.y.detach().numpy(), node_size=20, layout = 'kamada_kawai')
draw_graph_with_classes(g, classes = data.y.detach().numpy(), node_size=20, layout = 'circular')
draw_graph_with_classes(g, classes = data.y.detach().numpy(), node_size=20, layout = 'spectral')
draw_graph_with_classes(g, classes = data.y.detach().numpy(), node_size=20, layout = 'shell')
draw_graph_with_classes(g, classes = data.y.detach().numpy(), node_size=20, layout = 'random')

        # 'spring': nx.spring_layout,
        # 'kamada_kawai': nx.kamada_kawai_layout,
        # 'circular': nx.circular_layout,
        # 'spectral': nx.spectral_layout,
        # 'shell': nx.shell_layout,
        # 'random': nx.random_layout



print(torch.unique(data.y))

