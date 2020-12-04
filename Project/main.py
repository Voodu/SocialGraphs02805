# %%
import io
import string
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import requests

from bs4 import BeautifulSoup
from fa2 import ForceAtlas2
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.notebook import tqdm
from wordcloud import WordCloud

# %% [markdown]
# # Lord of the Rings - analysis
# In the project we would like to analyze text of three LotR books and movie scripts. Two main goals are to understand connections between characters and sentiment change over time in books and movies.
# 
# Analysis will be performed over all 300-word chunks of text from book/script. Movie scripts are taken from https://www.imsdb.com/ and books are taken from http://ae-lib.org.ua/. Character information and names list are taken from LotR wiki: https://lotr.fandom.com/wiki

# %% [markdown]
# ## Preparing the data
# After downloading the books and the scripts, a character list was needed. To get it, we combined https://lotr.fandom.com/wiki/Category:The_Lord_of_the_Rings_Characters list and sublists appearing in it. After cleaning the data, 174 characters remained in the final list. 
#
# The most common attribute on the wiki, which appeared on most of the pages, was character race. Therefore, it was added to each character information in the list.
#
# Three things happen in the following cells:
#
# 1. The data is loaded from CSV to `characters` dataframe
# 1. It is enriched with race information from the wiki
# 1. List of names is extracted to separate variable

# %%
# Just read the data from the cleaned CSV
characters = pd.read_csv('data/characters.csv', header=0, delimiter='\t', index_col=0)

# %%
# Enrich the data with race information
# CSS selector used to find race on the page
race_selector = '[data-source=race] > div > a'
# Dictionary fixing redirect links pointing to the same article,
# ex. Men is the same as Man
race_translations = {
    'Ainur': 'Ainur',
    'Black_Uruks': 'Black Uruks',
    'Black_Uruk': 'Black Uruks',
    'Dr%C3%BAedain': 'Druedain',
    'Dwarves': 'Dwarves',
    'Eagle': 'Eagles',
    'Eagles': 'Eagles',
    'Elves': 'Elves',
    'Ent': 'Ents',
    'Ents': 'Ents',
    'Great_Eagles': 'Eagles',
    'Half-elven': 'Half-elven',
    'Hobbit': 'Hobbits',
    'Hobbits': 'Hobbits',
    'Man': 'Men',
    'Men': 'Men',
    'Orc': 'Orcs',
    'Orcs': 'Orcs',
    'Skin-changer': 'Skin-changer',
    'Spiders': 'Spiders',
    'Theories_about_Tom_Bombadil': 'Unknown',
    'Uruk-hai': 'Uruk-hai'
}
races = set(race_translations.values())

def get_race(row):
    '''
    Extracts race information from the wiki about the character
    '''
    page_html = requests.get(row.link).text
    soup = BeautifulSoup(page_html, 'html.parser')
    link = soup.select_one(race_selector)
    if link is None: # sometimes it's 'Race' instead of 'race'
        link = soup.select_one(race_selector.replace('race', 'Race'))
    if link:
        race = race_translations[link['href'].split('/')[-1]]
    else:
        race = 'unknown'
    return race.lower()

def enrich_with_race(df, pickle_path='characters_race.p'):
    '''
    Enriches dataframe with race information. 
    If possible, reads from local file.
    '''
    try:
        return pickle.load(open(f'{pickle_path}', 'rb'))
    except Exception:
        print(f'Loading {pickle_path} failed. Recreating.')
    df['race'] = df.apply(get_race, axis=1)
    df = df[['race', 'link']]
    pickle.dump(df, io.open(f'{pickle_path}', 'wb'))
    return df

# Add column with race
characters = enrich_with_race(characters)

# %%
# Extract names list from the data
characters.index = characters.index.str.lower()
lotr_names = list(characters.index)
characters.head()

# %% [markdown]
# ## Building the graphs
# To allow analysis of various things, subgraphs for each book/movie were created and combined later. This allowed to analyze them separately and in groups, when necessary.
# ### Algorithm of graph building
# 1. For every data source DS:
#    1. Create empty graph G
#    1. Tokenize text of the given data source
#    1. Divide tokens into chunks, ex. [abcdefhgijk] and chunk size = 3 will create 3 chunks [abc], [def], [ghi], [jk] (where abc... are tokens)
#    1. For every chunk:
#       1. For every token, check if it is a valid character name and if so, add it to temporary list
#       1. Create complete graph (i.e. every node connected) from the temporary list
#       1. Combine chunk subgraph and G
#    1. Return G and add it subgraphs list
# 1. Combine subgraphs as needed, ex. create graph of all the books, movies or all the data sources
#
# 
# Every node of the graph is named after the character and holds its race as the attribute.

# %%
# Functions for text processing and graph building
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

def build_graph(filepath, names, chunk_size=300):
    '''
    Builds graph of characters with given names from data in specified file.
    Divides text in chunks of chunks_size and connects each character in that chunk.
    Afterwards combines all chunk-graphs together and returns the resulting graph.
    '''
    graph = nx.Graph()
    with io.open(filepath, 'r', encoding='utf8') as f:
        text = f.read()
    tokens = tokenize_text(text)
    for chunk in get_chunks(tokens, chunk_size):
        chunk_characters = [word for word in chunk if word in names]
        subgraph = nx.complete_graph(chunk_characters)
        graph = nx.compose(graph, subgraph)
    
    return graph

def set_race_(graph, df):
    '''Gives every node in the graph proper race information'''
    for name, fields in df.iterrows():
        if name in graph.nodes:
            graph.nodes[name]['race'] = fields.race

#%%
# Build the graphs from the declared datasources
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
    set_race_(subgraphs[-1], characters)

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

# %% [markdown]
# ## Initial analysis
# At the beginning, we wanted to analyze the global graph which connects books and movies.
# 
# The most basic statistics: 

# %%
print("Nodes:", lotr_graph.number_of_nodes())
print("Edges:", lotr_graph.number_of_edges())

# %% [markdown]
# The next step was to check, how the node degrees are distributed in the graph.

# %%
# Plotting the node distribution
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
barchart_distributions(degrees, title, 'Figure 1. Distribution of node degrees in LotR character graph connecting books and movies.', 'Node degree', 'Count')
plt.show()

# %% [markdown]
# And visualization of the whole graph itself. TODO: More comments

# %%
def get_node_size_map(graph):
    '''
    Returns size map taking node degree into account
    '''
    degrees = dict(graph.degree)
    return [v for v in degrees.values()]

def get_node_color_map(graph):
    '''
    Returns node color map taking race into account
    '''
    color_map = []
    for node in graph.nodes(data=True):
        color = '#00' + format(hash(node[1]['race'])%0xFFFF, 'x')
        color_map.append(color)
    return color_map

plt.figure(figsize=(14, 8))
# Determine node positions using Force Atlas 2 and draw. Use default config.
positions = ForceAtlas2().forceatlas2_networkx_layout(lotr_graph, pos=None, iterations=500)
nx.draw_networkx_nodes(
    lotr_graph,
    positions,
    node_size=get_node_size_map(lotr_graph),
    node_color=get_node_color_map(lotr_graph))
nx.draw_networkx_edges(
    lotr_graph,
    positions,
    width=0.1)
plt.axis('off')
caption = 'Figure 2. Visualization of the LotR graph'
plt.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=12)
plt.show()

# %% [markdown]
# TODO: Comments, intro to further analysis etc.

# %%
def report_characters(degrees):
    '''
    Prints information about 5 most connected characters according to the edge directions
    '''
    top = sorted(degrees, key=lambda x: x[1], reverse=True)[:5]
    print(f'Top 5 connected characters')
    print(f'{"name:": <35}links:')
    for h in top:
        print(f'{h[0]: <35}{h[1]}')


report_characters(lotr_graph.degree())
# %%
