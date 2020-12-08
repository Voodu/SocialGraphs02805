# %%
import io
import math
import os
import string
import random

import networkx as nx
import numpy as np
import pandas as pd
import pickle
import requests
import community
from functools import partial

from PIL import Image
from bs4 import BeautifulSoup
from fa2 import ForceAtlas2
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from wordcloud import WordCloud
import seaborn as sns

sns.set_theme()
sns.set_palette('gnuplot2')

tqdm = partial(tqdm, position=0, leave=True)
random.seed(42)
np.random.seed(42)

# %% [markdown]
# # Lord of the Rings - analysis
# %% [markdown]
# ## Motivation
# The project is about analyzing Lord of the Rings characters. By taking books and movie scripts we would like to understand connections between characters and check how does the sentiment in them change over time
#
# Analysis will be performed over all 300-word chunks of text from book/script. Movie scripts are taken from https://www.imsdb.com/ and books are taken from http://ae-lib.org.ua/. Character information and names list are taken from LotR wiki: https://lotr.fandom.com/wiki. It was chosen over normal wikipedia, because it has more comprehensive list of characters and their descriptions are structured very similarly (ex. it is easier to scrap race of every character).

# %% [markdown]
# ## Preparing the data & basic statistics
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
    except FileNotFoundError:
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
# ### Building the graphs
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
    pickle.dump(tokens, io.open(f'{filepath}.p', 'wb'))
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
    except FileNotFoundError:
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
# ### Introductory comments on the dataset
# As will be seen in the next section, the dataset is much smaller than the hero network created during the course. It is, however, hard to avoid - J.R.R. Tolkien had to create all the characters on his own, while Marvel and DC heroes are created by numerous writers.
#
# More comments, which take concrete analysis information into account, are provided below.


# %% [markdown]
# ## Analysis part 1 - Graphs and character network

# %% [markdown]
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
# That analysis will give us good understanding of the network, connections between nodes, and its overall shape. 
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
    return [v * 10 for v in degrees.values()]


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
# ### The node & edge count
# %% [markdown]
# #### Analysis
# Checking node and edge count will give us information about complexity of each network (and 'content richness' of each media type).

# %% [markdown]
# **Books graph**
basic_stats(books_graph)

# %% [markdown]
# **Movies graph**
basic_stats(movies_graph)

# %% [markdown]
# **Combined graph**
basic_stats(combined_graph)

# %% [markdown]
# #### Comments
# First of all, there's clear difference between number of nodes in books graph and movies graph. The former one has 149 nodes, while the latter one has only 62 of them. Obviously, it was easier for Tolkien to write about some character (or at least mention them) than for Jackson to find nearly 150 cast members (not including background actors). This leads to the conclusion that book is over 2 times more rich in content in terms of number of characters.
#
#  It is also interesting that combined graph contains 155 nodes, so 6 more than books. It means that some characters were created by movie director, even though they did not appear in the original book.
#
# Another conclusion from the number of nodes and edges is that one has to read the books and watch the movies to have some basic knowledge about every character in the Lord of the Rings universe.
#
# One can also notice that our names list has 173 entries, while combined graph has around 15 nodes less. The missing nodes are listed below:
set(lotr_names).difference(set(combined_graph.nodes))


# %% [markdown]
# Those nodes usually come from the movies and are non-canonical characters. They are not in the graph, because even in the movie script they were background characters and were not given any name. Articles in wiki probably use names created by fans or obtained from other sources (ex. director comments).

# %% [markdown]
# ### Node degree distribution
# %% [markdown]
# #### Analysis
# Degree distribution will allow to see, what type of network was created. 

# %% [markdown]
# **Books graph**
title = 'Degree distribution in book character graph'
caption = 'Figure 1. Distribution of node degrees in LotR character graph connecting books.'
nodes_distribution_barchart(books_graph, title, caption)

# %% [markdown]
# **Movies graph**
title = 'Degree distribution in book character graph'
caption = 'Figure 2. Distribution of node degrees in LotR character graph connecting movies.'
nodes_distribution_barchart(movies_graph, title, caption)

# %% [markdown]
# **Combined graph**
title = 'Degree distribution in combined character graph'
caption = 'Figure 3. Distribution of node degrees in LotR character graph connecting books and movies.'
nodes_distribution_barchart(combined_graph, title, caption)

# %% [markdown]
# #### Comments
# In general, one can think that the degree distributions follow the power-law, but it is not exactly the case here. For network to be scale-free, there should be big number of nodes with the lowest degree (i.e. 1) and it should exponentially decrease with the degree. Here, on every plot the highest count is around degree 10. It decreases later, but it does not happen smoothly. One can say, at most, that those networks are 'scale-free-like'. It makes perfect sense, though, as they are not typical social networks or semantic networks. They are artificially build using predefined chunk size. They may be considered as semi-social networks, as they are build using character interactions.
#
# It is also quite interesting that nodes in movie network have lower degree in general - they vary between 1 and around 60, while in books they go from 1 to around 120. This may be caused by the different nature of media - in a book a character can think about someone else who is not currently present with them. In a movie, there are no character thoughts and characters interact only with those present in the scene with them. Therefore, there is less 'mixing' of the characters, as they are contained in their scenes. Moreover, there are just more characters mentioned in the books, as seen in the node counts.

# %% [markdown]
# ### Most connected nodes
# %% [markdown]
# #### Analysis
# Most connected nodes will show us, what are the most important characters in the trilogy.

# %% [markdown]
# **Books graph**
report_characters(books_graph)

# %% [markdown]
# **Movies graph**
report_characters(movies_graph)

# %% [markdown]
# **Combined graph**
report_characters(combined_graph)

# %% [markdown]
# #### Comments
# The most connected nodes in all the graphs are partially similar, but not as much as one may assume. All of them include some of the most important characters like four hobbits, Gandalf or Aragorn. Books are more focused on hobbits, i.e. Frodo, Sam, Merry and Pippin, as they occupy top positions in the ranking, while movies are more focused on Aragorn, Legolas, Gimli, Gandalf and Frodo - the team one usually sees on the movie posters. The biggest surprise is very high degree of Bilbo in books. It may be caused by the fact, the he is uncle of Frodo who mentions him quite often in the book. In movie there is no way to show character's thoughts, so Bilbo is not as highly ranked as in the book.

# %% [markdown]
# ### Graph visualization
# %% [markdown]
# #### Analysis
# Visualizations will help to confirm previous results and also see, how the race of characters is distributed in the story.

# %% [markdown]
# **Books graph**
caption = 'Figure 4. Visualization of the Lord of the Rings graph (book characters)'
draw_fa2(books_graph, caption)

# %% [markdown]
# **Movies graph**
caption = 'Figure 5. Visualization of the Lord of the Rings graph (movie characters)'
draw_fa2(books_graph, caption)

# %% [markdown]
# **Combined graph**
caption = 'Figure 6. Visualization of the Lord of the Rings graph (books and movies combined)'
draw_fa2(combined_graph, caption)

# %% [markdown]
# #### Comments
# When it comes to visualizations, they are not as pretty as for the hero network created during the course. Even though nodes are colored according to the character races taken from wiki, one cannot find clear visual separation, communities or anything similar. This will be exposed only by using specific algorithms for communtiy detection in part 2.
#
# The graphs, however, confirm observations from the previous points - as books have more nodes and more connections between them, the graph looks way more dense than the movie one.
#
# Also, thanks to the coloring of nodes, it can be observed that, although there are many creature races in the book and movies, vast majority of the characters in the story are men or hobbits.

# %% [markdown]
# ## Analysis part 2 - Communities and sentiment
# ### Communities analysis

# In this section, the communities were found between the sources: the books, movies, and combined. This will help to visualise graphs again and analyse if the communities make sense according to our common knowledge of the topic.

# To identify communities, the *community* package with the method best_partition was used. In this approach, firstly, a dendrogram is generated using the Louvain algorithm. The algorithm calculates the relative densities between the edges inside of the communities with respect to the outside edges. This algorithm is not optimal to be used when considering all possible iterations, thus a heuristic approach is performed. To be sure that everytime the split be the same, the random seed at the top of the notebook was established. Then, the dendrogram is cut in the place of the highest partition, to obtain the highest modularity of the split.

# The next step was to name every community with a mostly connected node inside of each. This will help to distinguish communities during further analysis.


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
        comm = {key: value for key, value in attributes.items()
                if value == index}
        max_degree_node = max(dict(graph.subgraph(comm.keys()).degree()).items(),
                              key=lambda x: x[1])

        for node in comm.keys():
            graph.nodes[node]['community'] = max_degree_node[0]
    return graph


# %% [markdown]
# As it turned out, the number of communities is relatively similar between executions. For current seed, it is the same for books, movie which means that this is consistent behaviour between the sources. For the combined one, there are one more community, which may indicate that some of the characters met each other in different context between books and movies.

print('Movies communities: ')
movies_communities = find_communities_from(movies_graph)
movies_graph = set_community_attribute(movies_graph, movies_communities)

print('Books communities: ')
books_communities = find_communities_from(books_graph)
books_graph = set_community_attribute(books_graph, books_communities)

print('Combined communities: ')
combined_communities = find_communities_from(combined_graph)
combined_graph = set_community_attribute(combined_graph, combined_communities)


# %% [markdown]
# ### Community distribution
#
# Here, the community distributions were found to see, if the amounts of nodes in each community is balanced and check how it is distributed.
#
# To perform that, the binning of the amounts were performed.

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
    plt.figtext(0.5, -0.1, caption, wrap=True,
                horizontalalignment='center', fontsize=12)
    plt.show()


# %% [markdown]
# The sizes are similar between the sources. The interesting fact is that combined communities have fewer number of nodes in each community and it contains one more community than book and movie sources. Movie communities are the smallest because due to the much smaller amount of words, fewer characters were introduced explicitly. It seems that book communities fill the gap in the combined option.

caption = 'Figure {}. The histogram is representing the number of members\nassigned to each community from the {} community.'

[plot_community_distribution(comm[0], comm[1], comm[2]) for comm in [
    ('movies', movies_communities, caption.format(7, 'movies')),
    ('books', books_communities, caption.format(8, 'books')),
    ('combined', combined_communities, caption.format(9, 'combined'))
]]


# %% [markdown]
# ### Community graphs
#
# Another interesting analysis can be performed by plotting again the graphs but this time, depending on the community, instead of the race of characters.
#
# To do that, the very similar approach was performed but using a different property.

def get_community_nodes(graph, community_id):
    '''
    Return nodes of specific race in the graph
    '''

    nodes = [node for node, value in nx.get_node_attributes(graph, 'community').items() if
             value == community_id]
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


# %% [markdown]
# After plotting the graphs where nodes are coloured by the community, it can be seen that communities not only are bounded by the race of the characters but additionally by their relationship in the books and movies. This example could be presented by the connections for Bilbo. On the movie graph, this relation is much less significant than in the books which places Bilbo as more significant character in books than in movies. Also in books Bilbo has got much more connections inside of his community than in movies.

caption = 'Figure {}. Visualization of the Lord of the Rings communities graph ({} characters)'
[draw_community_fa2(comm[0], comm[1]) for comm in [
    (books_graph, caption.format(10, 'books')),
    (movies_graph, caption.format(11, 'movies')),
    (combined_graph, caption.format(12, 'combined')),
]]


# %% [markdown]
# ### TF-IDF Wordclouds

# Similarly to the previous project, the TF-IDF wordclouds were generated for each community. This time, there is no possibility to split the webpages between characters because all of them are mentioned in one long text. For this to work, the texts were again split into chunks. If a certain name of character were present in the chunk, the chunk was assigned to the community text. Then TF-IDF was calculated using those communities chunks. After that, the term frequencies were visualised using the wordclouds with a ring mask. The community chunks were created for both sources, however, the clouds were generated only for book. Both sets of community sources will be used later to present the community sentiment in books and movies.

# The following formula was used: $tf(t, d) * idf(t, D) = f_{t, d} * (log(N / 1+n_t) + 1)$. The smooth logarithmic function was chosen because of its slow descending curve. The weights in tf-idf filter out the common terms, hence the value for base defines the "speed" of filtering out those terms.


def get_pickled_tokens(source):
    '''
    Loads tokens from specified source file
    '''
    print('Loading pickled tokens')
    tokens = {}
    for filepath in [filepath for filepath in filepaths if source in filepath]:
        tokens[filepath] = pickle.load(io.open(f'{filepath}.p', 'rb'))

    return tokens


def create_community_text(tokens, graph, chunk_size=100):
    '''
    Returns dict with communities as keys and related text as values
    '''
    communities_dict = nx.get_node_attributes(graph, 'community')
    communities_texts = {key: [] for key in
                         set(nx.get_node_attributes(graph, 'community').values())}
    tokens_list = []
    for token_part_list in tokens.values():
        tokens_list += token_part_list

    for chunk in get_chunks(tokens_list, chunk_size):
        for comm in communities_dict:
            if comm in chunk:
                character = communities_dict[comm]
                communities_texts[character] += chunk

    return communities_texts


def calculate_tf_idf(dict_texts, source):
    '''
    Calculate tf-idf for each word in communities.
    '''
    try:
        print(f'tf_idf_{source}.p')
        return pickle.load(io.open(f'tf_idf_{source}.p', 'rb'))
    except FileNotFoundError:
        print('Loading failed. Recreating')
    tf_idf_communities = {}
    for index, current_text in tqdm(dict_texts.items()):
        tf_idf = FreqDist(current_text)
        other_texts = {key: dict_texts[key]
                       for key in dict_texts.keys() if key != index}

        for word in tqdm(tf_idf.keys()):
            counter = 0
            for _, other_text in other_texts.items():
                if word in other_text:
                    counter += 1
            tf_idf[word] = tf_idf[word] * (
                math.log(len(dict_texts.keys()) / (counter + 1), 10) + 1)

        tf_idf_communities[index] = tf_idf

    pickle.dump(tf_idf_communities, io.open(f'tf_idf_{source}.p', 'wb'))
    return tf_idf_communities


# %%

books_tokens = get_pickled_tokens('book')
movies_tokens = get_pickled_tokens('movie')

books_communities_texts = create_community_text(books_tokens, books_graph, 100)
movies_communities_texts = create_community_text(
    movies_tokens, movies_graph, 50)

books_tf_idf = calculate_tf_idf(books_communities_texts, 'book')
movies_tf_idf = calculate_tf_idf(movies_communities_texts, 'movie')


# %%

def create_wordcloud(text, cmap, caption, mask=None):
    '''
    Create a wordcloud based on a provided text.
    '''
    wordcloud = WordCloud(
        # max_font_size=40,
        collocations=False,
        background_color='#000000',
        colormap=cmap,
        mask=mask
    ).generate(text)
    plt.figure(dpi=300)
    plt.title(caption)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def create_texts_from_list(word_weights):
    '''
    From the dict {word: weight} create a number of words depending on
    the weight - weight is rounded up to int.
    'text': 1.75 -> ['text', 'text']
    '''
    text = []
    for word, weight in word_weights.items():
        text.append(f'{word} ' * math.ceil(weight))
    return ' '.join(text)


def create_communities_wordclouds(tf_idf, cmap, source, mask):
    '''
    Creates wordclouds based on provided TD-IDF
    '''
    for index in tf_idf.keys():
        text = create_texts_from_list(tf_idf[index])
        create_wordcloud(
            text, cmap, f'Wordcloud for {source} community: {index.title()}', mask)


# %% [markdown]
# For presentation purposes to show communities in one source, only the book communities were chosen to present wordlcouds. Not every wordcloud seems to be significant, however, for "Aragorn" the whole Fellowship of the Ring is mentioned. For "Saruman" words like "eye", "old", "Isengard", "orc" are present. For "Frodo" other hobbits are mentioned: "Bilbo", "Pippin", "Sam".

mask = np.array(Image.open(os.path.join('data', 'mask.png')))

create_communities_wordclouds(books_tf_idf, 'Wistia', 'books', mask)


# %% [markdown]
# ### Sentiment over time
#
# One of the main aims of the project was to analyse the sentiment over time for books and movies and compare them.
#
# This was done by finding a proper chunks size for both books and movies since movies have much fewer words than books. Moreover, the LabMT wordlist as used to calculate the mean sentiment per chunk. For each chunk, a weighted mean and *pandas* library were used for efficient computations. As it is actually a rolling sentiment which is calculated for both sources, there is no need to adjust them in time because both lines should be similarly aligned.


def calculate_rolling_sentiment(source, chunk_size=300):
    '''
    Calculating the sentiment for each of the character page and setting
    the value to the character node as an attribute.
    '''
    try:
        print(f'Loading {source}_{chunk_size}_sent.p')
        return pickle.load(io.open(f'{source}_{chunk_size}_sent.p', 'rb'))
    except FileNotFoundError:
        print('Loading failed. Recreating.')
    sentiment_url = 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0026752.s001&type=supplementary'
    sentiment_values = pd.read_csv(sentiment_url, skiprows=3, delimiter='\t')[
        ['word', 'happiness_average']]

    tokens = get_pickled_tokens(source)
    rolling_sentiment = pd.DataFrame()
    for part_index, part in enumerate(tokens.values()):
        rolling_sentiment_part = pd.DataFrame()
        for index, chunk in enumerate(get_chunks(part, chunk_size)):
            chunk_tokens = FreqDist(chunk)
            sentiment = pd.DataFrame.from_dict(
                chunk_tokens, orient='index').reset_index()
            sentiment.columns = ['word', 'count']
            sentiment['chunk'] = index
            sentiment = sentiment.merge(sentiment_values, on=['word'])
            sentiment['mean'] = (
                sentiment['happiness_average'] * sentiment['count']).sum() / sentiment['count'].sum()
            sentiment = sentiment[['chunk', 'mean']].groupby(
                'chunk').mean().reset_index()
            rolling_sentiment_part = pd.concat(
                [rolling_sentiment_part, sentiment])

        rolling_sentiment_part['part'] = part_index
        rolling_sentiment = pd.concat(
            [rolling_sentiment, rolling_sentiment_part])

    pickle.dump(rolling_sentiment, io.open(
        f'{source}_{chunk_size}_sent.p', 'wb'))
    return rolling_sentiment


# %%

book_sentiment = calculate_rolling_sentiment('book', 200)
movie_sentiment = calculate_rolling_sentiment('movie', 50)


# %%

def plot_sentiment(book_sentiment, movie_sentiment, caption):
    '''
    Creates book & sentiment plots with provied caption
    '''
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(caption)
    plots = [311, 312, 313]

    for index, plot in enumerate(plots):
        ax1 = fig.add_subplot(plot, label="1")

        ax1.set_title(f'Part {index+1}')
        book_set = book_sentiment.loc[book_sentiment['part'] == index]
        movie_set = movie_sentiment.loc[movie_sentiment['part'] == index]
        # create line plot of y1(x)
        line1 = ax1.plot(book_set['chunk'],
                         book_set['mean'], '#c71671', alpha=0.6)
        ax1.set_xlabel('Book chunk')
        ax1.set_ylabel('Sentiment')

        ax2 = ax1.twiny()

        # create line plot of y2(x)
        line2 = ax2.plot(movie_set['chunk'],
                         movie_set['mean'], '#169ec7', alpha=0.6)
        ax2.set_xlabel('Movie chunk')
        # add a legend, and position it on the upper right

        if index == 2:
            plt.legend((line1[0], line2[0]),
                       ('Book', 'Movie'), loc='lower right')

    plt.tight_layout()
    plt.show()


# %% [markdown]
#
# Below, the comparison for each part of the trilogy in both sources was presented. The movie sentiment fluctuates much more than the book one - probably because of the amount of words in each source. Nonetheless, the lines are relatively aligned, especially, especially during the climaxes at the end of first and third part of the trilogy. What is worth mentioning is the fact that there is no guarantee that fact in books and movies are presented at the same order. Moreover, the viewers feel the sentiment from the movie frames mainly, thus not every moment in the films might be properly interpreted by the average sentiment. The sudden drops in sentiment at the end are caused by small size of the last chunk - chunks are not evenly sized.

plot_sentiment(book_sentiment, movie_sentiment,
               'Sentiment over time for movie scripts and book texts.')

# %% [markdown]
# ### Communities sentiment
#
# The sentiment was also calculated for each of the community. It was creating using previously generated chunks using the same LabMT wordlist. The weighted average was calculated again for each community and was presented in the tables below.


def calculate_communities_sentiment(community_text):
    '''
    Calculates sentiment based on given community text
    '''
    sentiment_url = 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0026752.s001&type=supplementary'
    sentiment_values = pd.read_csv(sentiment_url, skiprows=3, delimiter='\t')[
        ['word', 'happiness_average']]

    sentiment_df = pd.DataFrame()
    for comm, words in community_text.items():
        chunk_tokens = FreqDist(words)
        sentiment = pd.DataFrame.from_dict(
            chunk_tokens, orient='index').reset_index()
        sentiment.columns = ['word', 'count']
        sentiment['community'] = comm
        sentiment = sentiment.merge(sentiment_values, on=['word'])
        sentiment['mean'] = (sentiment['happiness_average'] *
                             sentiment['count']).sum() / sentiment['count'].sum()
        sentiment = sentiment[['community', 'mean']].groupby(
            'community').mean().reset_index()
        sentiment_df = pd.concat([sentiment_df, sentiment])

    sentiment_df = sentiment_df.groupby(['community']).mean().reset_index()
    sentiment_df = sentiment_df.sort_values('mean', ascending=False)

    return sentiment_df


# %% [markdown]
# The happiest community throughout the books is "Frodo", probably because of the happy moments with his friends, positive beginning and ending of the story. The most negative is "Horn" where characters like Gollum, Muzgrash, Sagrat (orcs) indicates negative feelings during battles.

calculate_communities_sentiment(books_communities_texts)


# %% [markdown]
# As far as movies are concerned, the situation is very similar to the books. What is interesting, the saddest community is represented by Gandalf – despite his affiliation to this community, there are also many orcs and Gollum which presents negative impact on the sentiment for their chunks of texts.

calculate_communities_sentiment(movies_communities_texts)


# %% [markdown]
# ## Discussion
# Our results expose the most important differences and indicate biggest similarities between books and movies. While most of them are not surprising, some of them are thought provoking, ex. how big is the difference between number of characters in the books and movies.
#
# To provide more interesting network analysis, more manual work with extended domain knowledge is required. Unfortunately, wiki is not complete, i.e. there are very few common attributes as race for every character. If there were additional information (ex. conflict side), more interesting insights would be possible to derive. The above analysis still enabled to understand the differences between book and movies, but definitely can be extended by providing additional data. 

# %% [markdown]
# ## Contributions
# - Piotr Ładoński: gathering & cleaning the data, analysis of graphs in part 1
# - Paweł Darulewski: communities & sentiment analysis in part 2
