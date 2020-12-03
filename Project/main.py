# %%
import io
import string
import networkx as nx
import numpy as np
import pandas as pd

from fa2 import ForceAtlas2
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.notebook import tqdm
from wordcloud import WordCloud

# %%
characters = pd.read_csv('data/characters.csv', header=0, delimiter='\t')

# Extract names list from the data
characters.name = characters.name.str.lower()
lotr_names = list(characters.name)
characters.head()

# %%
def tokenize_text(text):
    '''
    Parsing given text: removing punctuation, creating tokens,
    setting to lowercase, removing stopwords, lemmatizing.
    '''
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def get_chunks(list, chunk_size): 
	'''Generates chunks from list'''
	for i in range(0, len(list), chunk_size):  
		yield list[i:i + chunk_size] 

# %%
def build_graph(filepath, names, chunk_size=300):
    '''
    Builds graph of characters with given names from data in specified file.
    Divides text in chunks of chunks_size and connects each character in that chunk.
    Afterwards combines all chunk-graphs together and returns the resulting graph.
    '''
    graph = nx.Graph()
    # Read text of file
    with io.open(filepath, 'r', encoding='utf8') as f:
        text = f.read()
    tokens = tokenize_text(text)
    # For each n-word chunk:
    for chunk in get_chunks(tokens, chunk_size):
        # Take each word
        # If it exists in characters, add it to chunk_characters
        chunk_characters = [word for word in chunk if word in names]
        # Create complete chunk subgraph from chunk_characters 
        subgraph = nx.complete_graph(chunk_characters)
        # Combine global and chunk graph
        graph = nx.compose(graph, subgraph)
    
    return graph

filepaths = [
	'data/book/part1.txt',
	'data/book/part2.txt',
	'data/book/part3.txt',
	'data/movie/part1.txt',
	'data/movie/part2.txt',
	'data/movie/part3.txt'
]
# Create subgraph for every file
subgraphs = []
for filepath in filepaths:
    subgraphs.append(build_graph(filepath, lotr_names))

# Create graph from book connections
books_graph = nx.Graph()
for subgraph in subgraphs[:3]:
    books_graph = nx.compose(books_graph, subgraph)

# Create graph from movie connections
movies_graph = nx.Graph()
for subgraph in subgraphs[3:]:
    movies_graph = nx.compose(movies_graph, subgraph)

# Compose one big graph connecting all the books and movies
lotr_graph = nx.compose(books_graph, movies_graph)

# %%
print("Nodes:", lotr_graph.number_of_nodes())
print("Edges:", lotr_graph.number_of_edges())

# %%
def barchart_distributions(data, title, caption, xlabel='', ylabel='', subplot=111):
    '''
    Wrapper around various pyplot configuration options
    '''
    plt.subplot(subplot)
    vector = list(range(np.min(data), np.max(data) + 2))
    graph_values, graph_bins = np.histogram(data, bins=vector)
    plt.bar(graph_bins[:-1], graph_values, width=.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=12)

degrees = [degree for _, degree in lotr_graph.degree()]
plt.figure(figsize=(14, 6))
title = 'Degree distribution in LotR character graph'
barchart_distributions(degrees, title, 'Figure 1. Distribution of degrees in LotR character graph.', 'Node degree', 'Count')
plt.show()

# %%
def get_node_size_map(graph):
    '''
    Returns size map taking node degree into account
    '''
    degrees = dict(graph.degree)
    return [v for v in degrees.values()]

plt.figure(figsize=(14, 8))
# Determine node positions using Force Atlas 2 and draw. Use default config.
positions = ForceAtlas2().forceatlas2_networkx_layout(lotr_graph, pos=None, iterations=500)
nx.draw_networkx_nodes(
    lotr_graph,
    positions,
    node_size=get_node_size_map(lotr_graph))
nx.draw_networkx_edges(
    lotr_graph,
    positions,
    width=0.1)
plt.axis('off')
plt.show()

# %%
def report_characters(degrees):
    '''
    Prints information about 5 most connected characters according to the edge directions
    '''
    top = sorted(degrees, key=lambda x: x[1], reverse=True)[:5]
    print(f'Top 5 connected characters')
    print(f'{"name:": <35}{dir} links:')
    for h in top:
        print(f'{h[0]: <35}{h[1]}')


report_characters(lotr_graph.degree())
# %%
