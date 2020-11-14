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
characters.head()
characters.name = characters.name.str.lower()
names = list(characters.name)

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

def get_chunks(list, size): 
	'''Generates chunks from list'''
	for i in range(0, len(list), size):  
		yield list[i:i + size] 

# %%
files = [
	'book/part1.txt',
	'book/part2.txt',
	'book/part3.txt',
	'movie/part1.txt',
	'movie/part2.txt',
	'movie/part3.txt'
]
# Create empty undirected graph
g = nx.Graph()

# For each file:
for file in files[:]:
	# Read text of file
	with io.open(f'data/{file}', 'r', encoding='utf8') as f:
		text = f.read()
	tokens = tokenize_text(text)
#	Create list with 300-word chunks
# 	For each chunk:
	for chunk in get_chunks(tokens, 300):
#		Take each word
#		If it exists in characters, add it to buffer
		buffer = [word for word in chunk if word in names]
# 		Create complete subgraph from buffer 
		subgraph = nx.complete_graph(buffer)
# 		Combine global and chunk graph nx.compose(A, B) 
		g = nx.compose(g, subgraph)

# %%
print("Nodes:", g.number_of_nodes())
print("Edges:", g.number_of_edges())

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
    # plt.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=12)

degrees = [degree for _, degree in g.degree()]
plt.figure(figsize=(14, 6))
title = 'Degree distribution in LotR character graph'
barchart_distributions(degrees, title, '', 'Node degree', 'Count')
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
positions = ForceAtlas2().forceatlas2_networkx_layout(g, pos=None, iterations=500)
nx.draw_networkx_nodes(
    g,
    positions,
    node_size=get_node_size_map(g))
nx.draw_networkx_edges(
    g,
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


report_characters(g.degree())
# %%
