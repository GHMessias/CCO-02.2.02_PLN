import torch
import pandas as pd
from utils.graph_eval import *

yake_graph = torch.load('logs/2025-07-07_08-52-34/graphs/yake.pt', weights_only=False)
embedding_graph = torch.load('logs/2025-07-07_08-52-34/graphs/embedding_word2vec.pt', weights_only = False)
ren_graph = torch.load('logs/2025-07-07_08-52-34/graphs/ren.pt', weights_only=False)

num_nodes = pd.read_csv('dataset/2-cleaned_gemini.tsv', sep = '\t').shape[0]


print('Yake Graph')
print(evaluate_graph(yake_graph.edge_index, num_nodes, yake_graph.y))
# print('assortatividade ',evaluate_graph(yake_graph.edge_index, num_nodes, yake_graph.y, metric='assortativity'))
# print('modularidade', evaluate_graph(yake_graph.edge_index, num_nodes, yake_graph.y, metric='modularity'))
# print('node degree ',evaluate_graph(yake_graph.edge_index, num_nodes, yake_graph.y, metric='node_degree'))



print('embedding_graph')
print(evaluate_graph(embedding_graph.edge_index, num_nodes, embedding_graph.y))
# print('assortatividade ',evaluate_graph(embedding_graph.edge_index, num_nodes, embedding_graph.y, metric='assortativity'))
# print('modularidade', evaluate_graph(embedding_graph.edge_index, num_nodes, embedding_graph.y, metric='modularity'))

print('REN Graph')
print(evaluate_graph(ren_graph.edge_index, num_nodes, ren_graph.y))
# print('assortatividade ',evaluate_graph(ren_graph.edge_index, num_nodes, ren_graph.y, metric='assortativity'))
# print('modularidade', evaluate_graph(ren_graph.edge_index, num_nodes, ren_graph.y, metric='modularity'))