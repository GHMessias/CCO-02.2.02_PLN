from utils.utils import *
from utils.graph_eval import evaluate_graph
from graph_gen.graph_gen import *
from torch_geometric.data import Data
from torch_geometric.nn import GAE
from models.gcn_model import GCN
from torch_geometric.transforms import RemoveDuplicatedEdges
from graph_visualization.graph_visualization import plot_igraph_community, general_degree_distribution, visualize_attributes


# Fazer o input por json

df = pd.read_csv('dataset/2-cleaned_gemini.tsv', sep = '\t')
df = df.sample(2)

y = df['label'].tolist()
torch.tensor(y)

# palavras_a_remover = carregar_palavras_remocao(args.correlated_words_txt_path)

# Remover as palavras altamente correlacionadas
# df = remover_palavras_de_textos(df, 'news', palavras_a_remover)

x = feature_gen(df, 'news')

model_name = 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
model = AutoModel.from_pretrained(model_name)
edge_index = embedding_sim_graphs(df = df, text_column='news', embedding='bert', model = model, tokenizer = tokenizer)

graph_data = Data(x = x, edge_index = edge_index, y = torch.tensor(df['label'].tolist()))

    # Removendo edges duplicados
if graph_data.edge_index != None:
    transform = RemoveDuplicatedEdges()
    graph_data = transform(graph_data)

torch.save(graph_data, f'bert_graph.pt')