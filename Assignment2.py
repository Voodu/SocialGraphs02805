#%%
import io
import json
import re
import string
from urllib.parse import quote
from urllib.request import urlopen

import community
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
from fa2 import ForceAtlas2
from wordcloud import WordCloud
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#%%
# Constants setup
urls = [
    'https://raw.githubusercontent.com/SocialComplexityLab/socialgraphs2020/master/files/marvel_characters.csv',
    'https://raw.githubusercontent.com/SocialComplexityLab/socialgraphs2020/master/files/dc_characters.csv'
]
universes = ['marvel', 'dc']

# Base query for all the upcoming queries
baseurl = 'https://en.wikipedia.org/w/api.php?'
action = 'action=query'
content = 'prop=revisions&rvprop=content'
dataformat = 'format=json'
formatversion = 'formatversion=2'
query_base = f'{baseurl}{action}&{content}&{dataformat}&{formatversion}&titles='

# %%
# Functions for getting and preparing the data
def prepare_data(df):
    '''
    Prepares the dataframe for further analysis.\n
    Drops rows with ';' in 'WikiLink' column.\n
    Drops rows with NaNs.\n
    Drops rows with duplicates in 'WikiLink' column.\n
    Adds column with URL-ready WikiLinks (i.e. undescores instead of spaces).
    '''
    df.loc[df['WikiLink'] == ';', 'WikiLink'] = np.NaN
    df = df.dropna()
    df = df.drop_duplicates('WikiLink')
    df['WikiLink'] = df['WikiLink'].str.replace(';', '')
    df['WikiLink_url'] = df['WikiLink'].str.replace(' ', '_')
    return df

def get_preprocessed_csv_data(urls, universes):
    '''
    Returns dictionary with:\n
    keys - universes\n
    values - dataframes with data from provided urls
    '''
    data = {}
    for i, univ in enumerate(universes):
        data[univ] = pd.read_csv(urls[i], sep='|', index_col=0, header=0, names=['CharacterName', 'WikiLink'])
        data[univ] = prepare_data(data[univ])
    return data

def get_wiki_page(title):
    '''Returns content of the wikipage with specified title or None if it does not exist'''
    query = f'{query_base}{quote(title)}' # quote escapes weird Unicode characters
    wikijson = json.loads(urlopen(query).read())
    try:
        return wikijson['query']['pages'][0]['revisions'][0]['content']
    except Exception:
        return None

def save_pages_from_df(df, dir='output'):
    '''Fetches all the wikipages from the dataframe and stores them as text files in the specified directory'''
    def create_file_from_wiki(row):
        filename = row['CharacterName']
        page = get_wiki_page(row['WikiLink_url'])
        if page is None:
            print('No page for', row['WikiLink_url'])
            return
        with io.open(f'{dir}/{filename}.txt', 'w', encoding='utf8') as f:
            f.write(page)
    df.apply(create_file_from_wiki, axis=1)

wikilink_re = re.compile(r'\[\[(.*?)\]\]')
def get_wikilinks(text):
    '''Return all wikilinks in a given text'''
    wikilinks = set()
    for l in wikilink_re.finditer(text):
        wikilinks.add(l[1].split('|')[0])
    return wikilinks

def filter_hero_links(wikilinks, hero_wikilinks):
    '''Returns intersection between wikilinks and hero wikilinks'''
    return set(wikilinks).intersection(set(hero_wikilinks))

#%%
# Functions for graph building
def add_heroes_to_graph(heroes, universe):
    '''Adds hero nodes with specified universe attribute to the hero graph'''
    hero_graph.add_nodes_from(heroes, universe=universe)

def add_edges_to_graph(source, nodes):
    '''Connects given source to all the nodes in the hero graph'''
    for node in nodes:
        hero_graph.add_edge(source, node)

def update_graph_with_hero(hero, links, universe):
    '''Updates hero graph with information about given hero and links.'''
    hero_graph.add_node(hero, universe=universe)
    for univ in universes:
        univ_links = data[univ]['WikiLink']
        linked_heroes = filter_hero_links(links, univ_links)
        add_heroes_to_graph(linked_heroes, univ)
        add_edges_to_graph(hero, linked_heroes)

def update_graph(row, universe):
    '''Updates hero graph with information from a row.
    Determines proper nodes & connections from corresponding text file.'''
    filename, current_hero  = row['CharacterName'], row['WikiLink']
    try:
        with io.open(f'{universe}/{filename}.txt', 'r', encoding='utf8') as f:
            page_content = f.read()
            wikilinks = get_wikilinks(page_content)
            update_graph_with_hero(current_hero, wikilinks, universe)
    except Exception:
        pass

def remove_isolates(graph):
    '''Removes isolated nodes from the graph'''
    graph.remove_nodes_from(list(nx.isolates(graph)))

def get_gcc(graph):
    '''Return subgraph with Giant Connected Component'''
    return graph.subgraph(sorted(nx.weakly_connected_components(graph), key=len, reverse=True)[0])

# %%
# Get data & build the hero graph for exercises
data = get_preprocessed_csv_data(urls, universes)
# for univ in universes: # TODO: UNCOMMENT!
# 	save_pages_from_df(data[univ], univ)

hero_graph = nx.DiGraph()
for univ in universes:
    data[univ].apply(update_graph, args=(univ,), axis=1)
remove_isolates(hero_graph)
hero_graph = get_gcc(hero_graph)

# %% [markdown]
# # Exercise 1: Visualize the network (from lecture 5) and calculate basic stats (from lecture 4). For this exercise, we assume that you've already generated the network of superheroes, report work on the giant connected component.
# Here's what you need to do:
# ## Exercise 1a: Stats (see lecture 4 for more hints)

# %% [markdown]
# ### What is the number of nodes in the network?
# ### What is the number of links?
print('Number of nodes in GCC:', len(hero_graph.nodes))
print('Number of links in GCC:', len(hero_graph.edges))
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# ### What is the number of links connecting Marvel and DC? Explain in your own words what those links mean?
def count_cross_universe_links(node, graph):
    '''Counts number of connections to other universe from a given node'''
    universe = node[1]['universe']
    return np.sum((1 if graph.nodes[n]['universe'] != universe else 0 for n in graph.neighbors(node[0])))

total_links = np.sum((count_cross_universe_links(n, hero_graph) for n in hero_graph.nodes(data=True)))

print('Number of cross-universe links', total_links)
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# ### Who are top 5 most connected characters? (Report results for in-degrees and out-degrees). Comment on your findings. Is this what you would have expected.
def report_heroes(degrees, dir):
    '''Prints information about 5 most connected heroes according to the edge directions'''
    top = sorted(degrees, key=lambda x: x[1], reverse=True)[:5]
    print(f'Top 5 connected heroes with {dir} links')
    print(f'{"name:": <35}{dir} links:')
    for h in top:
        print(f'{h[0]: <35}{h[1]}')

report_heroes(hero_graph.in_degree(), 'entering')
print()
report_heroes(hero_graph.out_degree(), 'exiting')
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# ### Plot the in and out-degree distributions.
# #### a) Explain what you observe?
# #### b) Explain why the in-degree distribution is different from the out-degree distribution?

gcc_in_degrees = [degree for _, degree in hero_graph.in_degree()]
gcc_out_degrees = [degree for _, degree in hero_graph.out_degree()]

def barchart_distributions(data, title, caption, xlabel='', ylabel='', subplot=111):
    '''Wrapper around various pyplot configuration options'''
    plt.subplot(subplot)
    vector = list(range(np.min(data), np.max(data) + 2))
    graph_values, graph_bins = np.histogram(data, bins=vector)
    plt.bar(graph_bins[:-1], graph_values, width=.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=12)

plt.figure(figsize=(14, 6))
caption = 'Figure 1. Histograms showing distributions of node degrees in\ngraph visualizing connections between Wikipedia pages of Marvel and DC heroes.'
title = 'In-degree distribution in GCC of Marvel & DC hero graph'
barchart_distributions(gcc_in_degrees, title, caption, 'Node degree', 'Count', 121)
title = 'Out-degree distribution in GCC of Marvel & DC hero graph'
barchart_distributions(gcc_out_degrees, title, caption, 'Node degree', 'Count', 122)
plt.show()
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# ### Compare the degree distribution to a random network with the same number of nodes and probability of connection $p$. Comment or your results.
size, p = hero_graph.order(), 0.002
random_network = nx.erdos_renyi_graph(size, p)
random_degrees = [degree for _, degree in random_network.degree()]
gcc_degrees = [degree for _, degree in hero_graph.degree()]
plt.figure(figsize=(14, 6))
caption = 'Figure 2. Comparison between node degree distributions of random network and hero network.\n'
title = f'Degree distribution in Erdös-Renyi network\nN={size}, p={p}'
barchart_distributions(random_degrees, title, caption,' Node degree', 'Count', 121)
title = f'Degree distribution in GCC of Marvel & DC hero graph'
barchart_distributions(gcc_degrees, title, caption, 'Node degree', 'Count', 122)
plt.show()
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# ## Exercise 1b: Visualization (see lecture 5 for more hints)
# ### Create a nice visualization of the total network.
# #### a) Color nodes according to universe
# #### b) Scale node-size according to degree
# #### c) Whatever else you feel like.
# #### d) If you can get it to work, get node positions based on the Force Atlas 2 algorithm
def get_node_color_map(graph):
    '''Returns node color map taking universe into account'''
    color_map = []
    for node in graph.nodes(data=True):
        color_map.append('red' if node[1]['universe'] == 'marvel' else 'black')
    return color_map

def get_node_size_map(graph):
    '''Returns size map taking node degree into account'''
    degrees = dict(graph.degree)
    return [v for v in degrees.values()]

def get_edge_color(graph, n1 ,n2):
    '''Return edge color between two nodes'''
    univ1, univ2 = graph.nodes[n1]['universe'], graph.nodes[n2]['universe']
    if univ1 == univ2 == 'marvel':
        return 'blue'
    if univ1 == univ2 == 'dc':
        return 'yellow'
    return 'green'

def get_edge_color_map(graph):
    '''Returns edge color map taking universe into account'''
    return [get_edge_color(graph, n1, n2) for n1, n2 in graph.edges]

# Create undirected graph, as it works better with Force Atlas 2
hero_undir = hero_graph.to_undirected()
plt.figure(figsize=(14,8))
# Determine node positions using Force Atlas 2 and draw. Use default config.
positions = ForceAtlas2().forceatlas2_networkx_layout(hero_undir, pos=None, iterations=500)
nx.draw_networkx_nodes(
            hero_undir,
            positions,
            node_size=get_node_size_map(hero_undir),
            node_color=get_node_color_map(hero_undir))
nx.draw_networkx_edges(
            hero_undir,
            positions,
            width=0.1,
            edge_color=get_edge_color_map(hero_undir))
plt.axis('off')
caption = \
'''
Figure 3. Visual representation of the GCC of the hero graph.
Marvel and DC nodes are respectively red and black.
Marvel-Marvel edges are blue, DC-DC are yellow and cross-universe are green.
'''
plt.figtext(0.5, 0, caption, wrap=True, horizontalalignment='center', fontsize=12)
plt.show()
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# # Exercise 2: Create your own version of the TF-TR word-clouds (from lecture 7). For this exercise we assume you know how to download and clean text from the wiki-pages.
# Here's what you need to do:
# ### That's it, really. The title says it all. Create your own version of the TF-TR word-clouds. Explain your process and comment on your results.
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# # Exercise 3: Find communities and create associated TF-IDF word clouds (from lecture 7 and 8). In this exercise, we assume that you have been able to find communities in your network. It's OK to only work on a single universe in this one.
# Here's what you need to do:

# %% [markdown]
# ### Explain briefly how you chose to identify the communities: Which algorithm did you use, how does it work?


def find_communities_from(graph):
    """
    Finding the communities, plotting them and printing the count.
    """
    partition = community.best_partition(graph)

    # drawing
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(graph)
    count = 0.
    for com in set(partition.values()):
        count = count + 1.
        list_nodes = [
            nodes for nodes in partition.keys() if partition[nodes] == com]
        nx.draw_networkx_nodes(
            graph, pos, list_nodes, node_size=20, node_color=str(count / size))

    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.show()

    print(f'No. of found communities: {count}')

    return partition


def get_subgraph_by_attribute(graph, attribute, value):
    """Getting the subgraph basing on universe attribute"""

    nodes = (
        node
        for node, node_data
        in graph.nodes(data=True)
        if node_data.get(attribute) == value
    )

    return graph.subgraph(nodes)

# %% [markdown]
# > ### Answer:
# To identify communities, the `community` package with the method `best_partition` was used. In this approach, firstly, a dendrogram is generated using the Louvain algorithm. The algorithm calculates the relative densities between the edges inside of the communities with respect to the outside edges. This algorithm is not optimal to be used when considering all possible iterations, thus a heuristic approach is performed. Then, the dendrogram is cut in the place of the highest partition, to obtain the highest modularity of the split.

# %% [markdown]
# ### How many communities did you find in total?


marvel_graph = get_subgraph_by_attribute(hero_undir, 'universe', 'marvel')
marvel_communities = find_communities_from(marvel_graph)

# %% [markdown]
# ### Plot the distribution of community sizes.


def plot_community_distribution(universe, communities):
    data = [value for _, value in communities.items()]
    title = f'The {universe} communities size distribution'
    caption = f'The histogram is representing the number of members assigned to each community from the {universe} community.'
    barchart_distributions(data, title, caption, 'Communities', 'Count')
    plt.show()

# %% [markdown]
# > ### Answer:


plot_community_distribution('Marvel', marvel_communities)

# %% [markdown]
# ### For the 5-10 largest communities, create TF-IDF based rankings of words in each community. There are many ways to calculate TF-IDF, explain how you've done it and motivate your choices.


def get_biggest_communities(communities, amount):
    """
    Return the nodes for given amount of biggest communities.
    """

    data = [value for _, value in communities.items()]
    vector = list(range(np.min(data), np.max(data) + 2))
    graph_values, _ = np.histogram(data, bins=vector)
    biggest_communities = np.argsort(-graph_values)[:amount]

    return {k: v for k, v in communities.items() if v in biggest_communities}


def tokenize_text(text):
    """
    Parsing given text: removing punctuation, creating tokens,
    setting to lowercase, removing stopwords, lemmatizing.
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def tokenize_texts(heroes, universe):
    texts = {}
    for hero in heroes:
        try:
            with io.open(f'{universe}/{hero}.txt', 'r', encoding='utf8') as f:
                page_content = f.read()
                page_content = tokenize_text(page_content)
                texts[hero] = page_content
        except FileNotFoundError:
            pass

    return texts


biggest_communities = get_biggest_communities(marvel_communities, 6)
tokenized_texts = tokenize_texts(biggest_communities.keys(), 'marvel')

# %% [markdown]
# > ### Answer:
# The logarithmic function was chosen because of its slow rising curve. The weights in tf-idf filter out the common terms, hence the value for base defines the "speed" of filtering out those terms. 

# %% [markdown]
# ### Create a word-cloud displaying the most important words in each community (according to TF-IDF). Comment on your results (do they make sense according to what you know about the superhero characters in those communities?)


def create_wordcloud(text):
    """
    Create a wordcloud based on a provided text.
    """
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# # Exercise 4: Analyze the sentiment of the communities (lecture 8). Here, we assume that you've successfully identified communities. Unlike above - we work all communities. It's still OK to work with data from a single universe. More tips & tricks can be found, if you take a look at Lecture 8's exercises.
# A couple of additional instructions you will need below;
# - We name each community by its three most connected characters.
# - Average the average sentiment of the nodes in each community to find a community level sentiment.
# 
# 
# Here's what you need to do:

# %% [markdown]
# ### Calculate and store sentiment for every single page.
# ### TODO: CODE
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# ### Create a histogram of all character's associated page-sentiments.
# ### TODO: CODE
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# ### What are the 10 characters with happiest and saddest pages?
# ### TODO: CODE
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# ### What are the three happiest communities?
# ### TODO: CODE
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# ### What are the three saddest communities?
# ### TODO: CODE
# %% [markdown]
# > ### Answer:
# > ### TODO

# %% [markdown]
# ### Do these results confirm what you can learn about each community by skimming the wikipedia pages?
# ### TODO: CODE
# %% [markdown]
# > ### Answer:
# > ### TODO
# %%
