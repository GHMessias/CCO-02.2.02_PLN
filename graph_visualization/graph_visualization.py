import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_igraph_community(graph, C, class_colors, output_file="plots/graph_plot.png"):
    """
    Plot a graph with nodes colored by class and edges styled based on class membership.
    
    Parameters:
    ----------
    g : igraph.Graph
        The graph object created with igraph.
    
    C : list or numpy.ndarray
        Class labels for each node.
    
    class_colors : dict
        A dictionary where keys are class labels (0, 1, ...) and values are corresponding colors.
    
    output_file : str, default="graph_plot.png"
        Filename for saving the plotted graph.
    """

    g = graph.copy()

    # 1️⃣ Identify vertices with no edges (degree 0)
    isolated_vertices = [v.index for v in g.vs if g.degree(v.index) == 0]

    # 2️⃣ Remove isolated vertices
    g.delete_vertices(isolated_vertices)

    # 2️⃣ Basic setup
    N = len(C)
    num_classes = len(set(C))
    
    # 3️⃣ Assign node colors based on class labels using provided class_colors
    vertex_colors = [class_colors[class_id] for class_id in C]

    vertex_sizes = [10] * N

    # 6️⃣ Assign edge colors and weights based on whether edge is within the same class
    ecolors = []
    eweights = []
    for e in g.es:
        src, tgt = e.tuple
        if C[src] == C[tgt]:
            ecolors.append(class_colors[C[src]])  # Same color for same class edges
            eweights.append(2)
        else:
            ecolors.append("#000000")  # Black for inter-class edges
            eweights.append(1)
    
    # Assign edge weights as attributes (igraph uses this for layout)
    g.es["weight"] = eweights
    
    # 7️⃣ Finalize visual style dictionary
    visual_style = {
        "vertex_size": vertex_sizes,
        "vertex_color": vertex_colors,
        "vertex_label": None,
        "edge_width": eweights,
        "edge_color": ecolors,
        # "layout": g.layout_spectral(),
        "layout": g.layout_fruchterman_reingold(),
        "bbox": (1000, 1000),
        "margin": 20
    }
    
    # 8️⃣ Plot and save
    ig.plot(
        g,
        target=output_file,
        **visual_style
    )
    
    print(f"Graph saved as '{output_file}'")


def class_degree_distribution(g, C, class_colors, max_degree_threshold=50, save_path='plots/node_degree_plot.png'):
    """
    Plota a densidade de distribuição de grau (KDE) para cada classe no grafo gerado.
    Conta como vizinhos somente os nós da mesma classe.
    Ignora graus acima de max_degree_threshold.
    Se 'save_path' for fornecido, salva a imagem no caminho especificado.

    Args:
        g (igraph.Graph): Grafo do tipo igraph.
        C (list ou np.array): Vetor de classes dos nós, tamanho n.
        class_colors (dict): Dicionário com chave = classe e valor = cor associada.
        max_degree_threshold (int): Valor máximo de grau a ser considerado no gráfico.
        save_path (str, opcional): Caminho para salvar a imagem gerada. Se None, a imagem será apenas exibida.
    """ 
    # Verificação do tamanho
    n = len(C)
    assert len(g.vs) == n, "O número de nós no grafo deve ser igual ao tamanho de C."
    
    # Atribui classes ao grafo
    g.vs["class"] = C
    
    # Calcula graus por classe (somente entre nós da mesma classe)
    degree_distribution = {}
    for class_id in set(C):
        nodes_in_class = [i for i, cl in enumerate(C) if cl == class_id]
        degrees = []
        for node in nodes_in_class:
            degree = sum(1 for neighbor in g.neighbors(node) if C[neighbor] == class_id)
            if degree <= max_degree_threshold:
                degrees.append(degree)
        degree_distribution[class_id] = degrees

    # Verifica se há dados
    all_degrees = [d for degrees in degree_distribution.values() for d in degrees]
    if not all_degrees:
        print("Nenhum nó com grau abaixo do threshold foi encontrado.")
        return

    # Plotagem com KDE
    plt.figure(figsize=(10, 6))
    for class_id, degrees in sorted(degree_distribution.items()):
        if len(degrees) > 1:  # KDE precisa de pelo menos 2 pontos
            sns.kdeplot(degrees, label=f'Classe {class_id}',
                        color=class_colors[class_id], fill=True, alpha=0.3)

    plt.xlabel("Grau")
    plt.ylabel("Densidade Estimada")
    plt.title(f"Densidade de Grau por Classe (graus ≤ {max_degree_threshold})")
    plt.grid(True)
    plt.legend()

    # Salva ou mostra
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Imagem salva em: {save_path}")
    else:
        plt.show()

def general_degree_distribution(g, C, class_colors, max_degree_threshold=50, save_path='plots/general_node_degree_plot.png'):
    """
    Plota a densidade de distribuição de grau (KDE) para cada classe no grafo,
    considerando todas as conexões, não apenas as intra-classe.

    Args:
        g (igraph.Graph): Grafo do tipo igraph.
        C (list ou np.array): Vetor de classes dos nós, tamanho n.
        class_colors (dict): Dicionário com chave = classe e valor = cor associada.
        max_degree_threshold (int): Grau máximo considerado no gráfico.
        save_path (str): Caminho para salvar o gráfico. Se None, exibe na tela.
    """
    n = len(C)
    assert len(g.vs) == n, "O número de nós no grafo deve ser igual ao tamanho de C."

    g.vs["class"] = C

    # Obtém os graus de todos os nós
    all_degrees = g.degree()

    # Agrupa os graus por classe
    degree_distribution = {class_id: [] for class_id in set(C)}
    for idx, class_id in enumerate(C):
        degree = all_degrees[idx]
        if degree <= max_degree_threshold:
            degree_distribution[class_id].append(degree)

    # Verifica se há dados
    if not any(degree_distribution.values()):
        print("Nenhum grau dentro do limite especificado foi encontrado.")
        return

    # Plotagem com KDE
    plt.figure(figsize=(10, 6))
    for class_id, degrees in sorted(degree_distribution.items()):
        if len(degrees) > 1:  # KDE precisa de pelo menos 2 pontos
            sns.kdeplot(degrees, label=f'Classe {class_id}',
                        color=class_colors[class_id], fill=True, alpha=0.3)

    plt.xlabel("Grau")
    plt.ylabel("Densidade Estimada")
    plt.title(f"Distribuição Geral de Grau por Classe (graus ≤ {max_degree_threshold})")
    plt.grid(True)
    plt.legend()

    # Salva ou mostra
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Imagem salva em: {save_path}")
    else:
        plt.show()

def visualize_attributes(X, C, class_colors, save_path='plots/att_plot.png'):
    """
    Visualiza os atributos dos nós no grafo.
    Se a dimensionalidade de X for maior que 2, aplica t-SNE para reduzir para 2D.
    Plota os pontos com cores baseadas nas classes dos nós.

    Args:
        X (np.array): Matriz de atributos (n x d), onde n é o número de nós e d é o número de atributos.
        C (list ou np.array): Vetor de classes dos nós, tamanho n.
        class_colors (dict): Dicionário de cores, onde a chave é a classe e o valor é a cor associada.
        save_path (str, opcional): Caminho para salvar a imagem gerada. Se None, a imagem será apenas exibida.
    """
    # Se a dimensionalidade de X for maior que 2, aplica t-SNE para reduzir a 2D
    if X.shape[1] > 2:
        # Normalizar os dados antes de aplicar o t-SNE
        X_scaled = StandardScaler().fit_transform(X)

        # Aplica t-SNE para reduzir a dimensionalidade para 2
        tsne = TSNE(n_components=2, random_state=42)
        X_2D = tsne.fit_transform(X_scaled)
    else:
        # Se já for 2D, só usa X diretamente
        X_2D = X

    # Criar o gráfico
    plt.figure(figsize=(8, 6))

    # Plotando os pontos com cores baseadas nas classes
    for class_id, color in class_colors.items():
        # Seleciona os pontos que pertencem à classe `class_id`
        class_points = X_2D[np.array(C) == class_id]
        plt.scatter(class_points[:, 0], class_points[:, 1], color=color, label=f'Classe {class_id}', alpha=1, s = 15)

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Visualização dos Atributos com t-SNE ou 2D")
    plt.legend()
    plt.grid(True)

    # Se um caminho for fornecido, salva a imagem
    if save_path:
        plt.savefig(save_path)
        print(f"Imagem salva em: {save_path}")
    else:
        plt.show()