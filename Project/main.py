# %%
import io
import string
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import requests
import community

from bs4 import BeautifulSoup
from fa2 import ForceAtlas2
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.notebook import tqdm
from wordcloud import WordCloud
import seaborn as sns

sns.set_theme()
sns.set_palette('gnuplot2')

# %% [markdown]
# # Lord of the Rings - analysis
# %% [markdown]
# ## Motivation
# The project is about analyzing Lord of the Rings characters. By taking books and movie scripts we would like to understand connections between characters and check how does the sentiment in them change over time
#
# Analysis will be performed over all 300-word chunks of text from book/script. Movie scripts are taken from https://www.imsdb.com/ and books are taken from http://ae-lib.org.ua/. Character information and names list are taken from LotR wiki: https://lotr.fandom.com/wiki. It was chosen over normal wikipedia, because it has more comprehensive list of characters and their descriptions are structured very similarly (ex. it is easier to scrap race of every character).

# %% [markdown]
# ## Part 1 - graphs and character network
# %% [markdown]
# ### Preparing the data & basic statistics
# After downloading the books and the scripts, a character list was needed. To get it, we combined https://lotr.fandom.com/wiki/Category:The_Lord_of_the_Rings_Characters list and sublists appearing in it. After cleaning the data, 173 characters remained in the final list.
#
# Cleaning included i.a.:
# - removing non-character characters, ex. `The Ring` which is not a real character,
# - removing/shortening characters with multiple-word names like Samwise Gamgee to Sam. It had two reasons - first of all, tokenization works correctly only with single words, secondly some character full names are not used in the books/movies (ex. above Sam/Samwise),
# - removing duplicate characters - fortunately very rarely, but some characters were listed twice (ex. young/old version of them). We were not interested in this distinction.
#
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
characters = pd.read_csv('data/characters.csv',
                         header=0, delimiter='\t', index_col=0)

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
    if link is None:  # sometimes it's 'Race' instead of 'race'
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
# #### Building the graphs
# To allow analysis of various things, subgraphs for each book/movie were created and combined later. This allowed to analyze them separately and in groups, when necessary.
# ##### Algorithm of graph building
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
    tokens = [
        token for token in tokens if token not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def get_chunks(list, chunk_size):
    '''Generates chunks from list'''
    for i in range(0, len(list), chunk_size):
        yield list[i:i + chunk_size]


def remove_isolates(graph):
    '''
    Removes isolated nodes from the graph
    '''
    graph.remove_nodes_from(list(nx.isolates(graph)))


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


def build_subgraphs(filepaths, pickle_path='subgraphs.p'):
    '''
    Calls build_graph for every file in filepaths.
    Tries to read from local file if possible.
    '''
    try:
        return pickle.load(open(f'{pickle_path}', 'rb'))
    except Exception:
        print(f'Loading {pickle_path} failed. Recreating.')
    subgraphs = []
    for filepath in filepaths:
        subgraphs.append(build_graph(filepath, lotr_names))
        remove_isolates(subgraphs[-1])
        set_race_(subgraphs[-1], characters)
    pickle.dump(subgraphs, io.open(f'{pickle_path}', 'wb'))
    return subgraphs


def set_race_(graph, df):
    '''Gives every node in the graph proper race information'''
    for name, fields in df.iterrows():
        if name in graph.nodes:
            graph.nodes[name]['race'] = fields.race


# %%
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
subgraphs = build_subgraphs(filepaths)

# Create graph from book connections
books_graph = nx.Graph()
for subgraph in subgraphs[:3]:
    books_graph = nx.compose(books_graph, subgraph)

# Create graph from movie connections
movies_graph = nx.Graph()
for subgraph in subgraphs[3:]:
    movies_graph = nx.compose(movies_graph, subgraph)

# Compose one big graph connecting all the books and movies
combined_graph = nx.compose(books_graph, movies_graph)

# %% [markdown]
# #### Introductory comments on the dataset
# As will be seen in the next section, the dataset is much smaller than the hero network created during the course. It is, however, hard to avoid - J.R.R. Tolkien had to create all the characters on his own, while Marvel and DC heroes are created by numerous writers. 
#
# More comments, which take concrete analysis information into account, are provided below. 

# %% [markdown]
# ### Graph analysis
# We decided to perform analysis on several graph combinations and compare the results:
# - books graph
# - movies graph
# - combined graph
#
#
# For every situation we will:
# - check node & edge count
# - see node degree distribution
# - find most connected nodes
# - plot the network
#
#
# That analysis will give us good understanding of the network, connections between nodes, and its overall shape. Checking node and edge count will give us information about complexity of each network (and 'content richness' of each media type). Degree distribution will allow to see, what type of network was created. Most connected nodes will show us, what are the most important characters in the trilogy. Visualizations will help to confirm previous results and also see, how the race of characters is distributed in the stroy.
#
# Below we define functions used later for every graph.

# %%


def basic_stats(graph):
    '''
    Prints count of graph nodes and edges.
    '''
    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())


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
    plt.figtext(
        0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=12)


def nodes_distribution_barchart(graph, title, caption):
    '''
    Draws barchart with node distribution in the graph
    '''
    degrees = [degree for _, degree in graph.degree()]
    plt.figure(figsize=(14, 6))
    barchart_distributions(degrees, title, caption, 'Node degree', 'Count')
    plt.show()


def report_characters(graph):
    '''
    Prints information about 10 most connected characters
    '''
    degrees = graph.degree()
    top = sorted(degrees, key=lambda x: x[1], reverse=True)[:10]
    print(f'Top 10 connected characters')
    print(f'{"name:": <35}links:')
    for h in top:
        print(f'{h[0]: <35}{h[1]}')


def get_node_size_map(graph, nodes):
    '''
    Returns size map taking node degree into account
    '''
    degrees = dict(graph.degree(nodes))
    return [v*10 for v in degrees.values()]


def get_race_nodes(graph, race):
    '''
    Return nodes of specific race in the graph
    '''
    nodes = []
    for node in graph.nodes(data=True):
        if node[1]['race'] == race:
            nodes.append(node[0])
    return nodes


def get_property_color(prop):
    '''
    Return node color of the race
    '''
    return '#' + format(hash(prop) % 0xFFFFFF, 'x').zfill(6)


def draw_race_nodes(graph, race, positions):
    '''
    Draws nodes of the given race from the graph
    '''
    race_nodes = get_race_nodes(graph, race)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=race_nodes,
        node_size=get_node_size_map(graph, race_nodes),
        node_color=get_property_color(race),
        label=race,
        alpha=0.6
    )


def draw_fa2(graph, caption):
    '''
    Draws graph using Force Atlas 2
    '''
    plt.figure(figsize=(14, 8))
    # Determine node positions using Force Atlas 2 and draw. Use default config.
    positions = ForceAtlas2().forceatlas2_networkx_layout(
        graph, pos=None, iterations=500)
    for race in races:
        draw_race_nodes(graph, race.lower(), positions)
    nx.draw_networkx_edges(graph, positions, width=0.1, alpha=0.5)
    nx.draw_networkx_labels(graph, positions, font_size=7)
    plt.axis('off')
    plt.gca().set_facecolor('white')
    plt.figtext(0.5, -0.1, caption, wrap=True,
                horizontalalignment='center', fontsize=12)
    plt.legend(scatterpoints=1)
    plt.show()


# %% [markdown]
# #### Books graph
# **The node & edge count:**
basic_stats(books_graph)

# %% [markdown]
# **Node degree distribution:**
title = 'Degree distribution in book character graph'
caption = 'Figure 1. Distribution of node degrees in LotR character graph connecting books.'
nodes_distribution_barchart(books_graph, title, caption)

# %% [markdown]
# **Most connected nodes**
report_characters(books_graph)

# %% [markdown]
# **Graph visualization.**
caption = 'Figure 2. Visualization of the Lord of the Rings graph (book characters)'
draw_fa2(books_graph, caption)

# %% [markdown]
# #### Movies graph
# **The node & edge count:**
basic_stats(movies_graph)

# %% [markdown]
# **Node degree distribution:**
title = 'Degree distribution in book character graph'
caption = 'Figure 3. Distribution of node degrees in LotR character graph connecting movies.'
nodes_distribution_barchart(movies_graph, title, caption)

# %% [markdown]
# **Most connected nodes**
report_characters(movies_graph)

# %% [markdown]
# **Graph visualization.**
caption = 'Figure 4. Visualization of the Lord of the Rings graph (movie characters)'
draw_fa2(movies_graph, caption)

# %% [markdown]
# #### Combined graph
# **The node & edge count:**
basic_stats(combined_graph)

# %% [markdown]
# **Node degree distribution:**
title = 'Degree distribution in combined character graph'
caption = 'Figure 5. Distribution of node degrees in LotR character graph connecting books and movies.'
nodes_distribution_barchart(combined_graph, title, caption)

# %% [markdown]
# **Most connected nodes**
report_characters(combined_graph)

# %% [markdown]
# **Graph visualization.**
caption = 'Figure 6. Visualization of the Lord of the Rings graph (books and movies combined)'
draw_fa2(combined_graph, caption)

# %% [markdown]
# #### Comments and discussion on the results of graph analysis
# %% [markdown]
# ##### Number of nodes and edges
# There are several observations which can be made from the results.
#
# First of all, there's clear difference between number of nodes in books graph and movies graph. The former one has 149 nodes, while the latter one has only 62 of them. Obviously, it was easier for Tolkien to write about some character (or at least mention them) than for Jackson to find nearly 150 cast members (not including background actors). This leads to the conclusion that book is over 2 times more rich in content in terms of number of characters.
#
#  It is also interesting that combined graph contains 155 nodes, so 6 more than books. It means that some character were created by move director, even though they did not appear in the original book.
#
# Another conclusion from the number of nodes and edges is that one has to read the books and watch the movies to have some basic knowledge about every character in the Lort of the Rings universe.
#
# One can also notice that our names list has 173 entries, while combined graph has around 15 nodes less. The missing nodes are listed below:
set(lotr_names).difference(set(combined_graph.nodes))

# %% [markdown]
# Those nodes usually come from the movies and are non-canonical characters. They are not in the graph, because even in the movie script they were background characters and were not given any name. Articles in wiki probably use names created by fans or obtained from other sources (ex. director comments).

# %% [markdown]
# ##### Degree distribution
# In general, one can think that the degree distributions follow the power-law, but it is not exactly the case here. For network to be scale-free, there should be big number of nodes with the lowest degree (i.e. 1) and it should exponentially decrease with the degree. Here, on every plot the highest count is around degree 10. It decreases later, but it does not happen smoothly. One can say, at most, that those networks are 'scale-free-like'. It makes perfect sense, though, as they are not typical social networks or semantic networks. They are artificially build using predefined chunk size. They may be considered as semi-social networks, as they are build using character interactions.
#
# It is also quite interesting that nodes in movie network have lower degree in general - they vary between 1 and around 60, while in books they go from 1 to around 120. This may be caused by the different nature of media - in a book a character can think about someone else who is not currently present with them. In a movie, there are no character thoughts and characters interact only with those present in the scene with them. Therefore, there is less 'mixing' of the characters, as they are contained in their scenes. Moreover, there are just more characters mentioned in the books, as seen in the node counts. 

# %% [markdown]
# ##### Most connected nodes
# The most connected nodes in all the graphs are partially similar, but not as much as one may assume. All of them include some of the most important characters like four hobbits, Gandalf or Aragorn. Books are more focused on hobbits, i.e. Frodo, Sam, Merry and Pippin, as they occupy top positions in the ranking, while movies are more focused on Aragorn, Legolas, Gimli, Gandalf and Frodo - the team one usually sees on the movie posters. The biggest surprise is very high degree of Bilbo in books. It may be caused by the fact, the he is uncle of Frodo who mentions him quite often in the book. In movie there is no way to show character's thoughts, so Bilbo is not as highly ranked as in the book.
# %% [markdown]
# ##### Graphs visualization
# When it comes to visualizations, they are not as pretty as for the hero network created during the course. Even though nodes are colored according to the character races taken from wiki, one cannot find clear separation, communities or anything similar. This is caused by the way of narration in the books or movies - everything was happening simultaneously and characters were mixed together. There was no 'Frodo point of view' VS 'Sauron point of view', which would help in separating the opposite sides of the conflict.
#
# The graphs, however, confirm observations from the previous points - as books have more nodes and more connections between them, the graph looks way more dense than the movie one. 
#
# Also, thanks to the coloring of nodes, it can be observed that, although there are many creature races in the book and movies, vast majority of the characters in the story are men or hobbits. 

# %% [markdown]
# ##### Other thoughts
# To provide more interesting network analysis, more manual work with extended domain knowledge is required. Unfortunately, wiki is not complete, i.e. there are very few common attributes as race for every character. If there were additional information (ex. conflict side), more interesting insights would be possible to derive. The above analysis still enabled to understand the differences between book and movies, but definitely can be extended by providing additional data. Also, it was a bit disappointing that there were no clear hubs in the graph which would indicate different groups, ex. 'the ring team', wizards, or anything else. Most probably it was caused by the way the analysis was performed. It may be worth checking, how the network would look if it was created using wiki article analysis or by making a weighted edges and increasing the weight each time two characters are connected. Hopefully, that approach can help in finding reasonable communities.

# %% [markdown]
# ## Contributions
# - Part 1 was done by Piotr Ładoński
# - Part 2 was done by Paweł Darulewski
# %%


def find_communities_from(graph):
    '''
    Finding the communities, plotting them and printing the count.
    '''
    partition = community.best_partition(graph)
    count = len(set(partition.values()))

    print(f'No. of found communities: {count}')

    return partition


def set_community_attribute(graph, attributes):
    '''
    Function for setting the community attribute to the character node.
    '''

    for index in set(attributes.values()):
        comm = {key: value for key, value in attributes.items() if value == index}
        max_degree_node = max(dict(graph.subgraph(comm.keys()).degree()).items(), key=lambda x: x[1])

        for node in comm.keys():
            graph.nodes[node]['community'] = max_degree_node[0]
    return graph


# %%
print('Movies communities: ')
movies_communities = find_communities_from(movies_graph)
movies_graph = set_community_attribute(movies_graph, movies_communities)

print('Books communities: ')
books_communities = find_communities_from(books_graph)
books_graph = set_community_attribute(books_graph, books_communities)

print('Combined communities: ')
combined_communities = find_communities_from(combined_graph)
combined_graph = set_community_attribute(combined_graph, combined_communities)

# %%


def plot_community_distribution(source, communities, caption):
    '''
    Plot the distribution of the communities in the graph.
    '''

    data = [value for _, value in communities.items()]
    title = f'The {source} communities size distribution'
    values, bins = np.histogram(data, 11)
    plt.bar(bins[:-1], values, width=0.5)
    plt.title(title)
    plt.xlabel('Bin')
    plt.ylabel('Count')
    plt.grid()
    plt.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

# %%

caption = 'Figure {}. The histogram is representing the number of members\nassigned to each community from the {} community.'

[plot_community_distribution(comm[0], comm[1], comm[2]) for comm in [
    ('movies', movies_communities, caption.format(7, 'movies')),
    ('books', books_communities, caption.format(8, 'books')),
    ('combined', combined_communities, caption.format(9, 'combined'))
]]

# %%


def get_community_nodes(graph, community_id):
    '''
    Return nodes of specific race in the graph
    '''

    nodes = [node for node, value in nx.get_node_attributes(graph, 'community').items() if value == community_id]
    return graph.subgraph(nodes)


def draw_community_nodes(graph, community_id, positions):
    '''
    Draws nodes of the given community from the graph
    '''
    nodes = get_community_nodes(graph, community_id)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=nodes,
        node_size=get_node_size_map(graph, nodes),
        node_color=get_property_color(str(community_id)),
        label=community_id,
        alpha=0.5,
    )


def draw_community_fa2(graph, caption):
    '''
    Draws graph using Force Atlas 2
    '''
    plt.figure(figsize=(14, 8), dpi=300)
    # Determine node positions using Force Atlas 2 and draw. Use default config.
    positions = ForceAtlas2().forceatlas2_networkx_layout(
        graph, pos=None, iterations=500)
    for community_id in nx.get_node_attributes(graph, 'community').keys():
        draw_community_nodes(graph, community_id, positions)
    nx.draw_networkx_edges(graph, positions, width=0.1, alpha=0.4)
    nx.draw_networkx_labels(graph, positions, font_size=7)
    plt.axis('off')
    plt.figtext(
        0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.legend(scatterpoints=1, labelspacing=1)
    plt.show()


caption = 'Figure {}. Visualization of the Lord of the Rings communities graph ({} characters)'
[draw_community_fa2(comm[0], comm[1]) for comm in [
    (movies_graph, caption.format(10, 'movies')),
    (books_graph, caption.format(11, 'books')),
    (combined_graph, caption.format(12, 'combined')),
]]
