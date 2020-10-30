#%%
import networkx as nx
from fa2 import ForceAtlas2
import pandas as pd
import numpy as np

from urllib.request import urlopen
from urllib.parse import quote
import json
import io

import matplotlib.pyplot as plt

#%%
urls = ['https://raw.githubusercontent.com/SocialComplexityLab/socialgraphs2020/master/files/marvel_characters.csv',
 'https://raw.githubusercontent.com/SocialComplexityLab/socialgraphs2020/master/files/dc_characters.csv']
data = {}
univs = ['marvel', 'dc']

# %%
def clean_data(df):
	'''
	Cleans up the data in the dataframe.\n
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

# %%
for i in range(2):
	univ = univs[i]
	data[univ] = pd.read_csv(urls[i], sep='|', index_col=0, header=0, names=['CharacterName', 'WikiLink'])
	data[univ] = clean_data(data[univ])

# %%
# Base query for all the upcoming queries
baseurl = 'https://en.wikipedia.org/w/api.php?'
action = 'action=query'
content = 'prop=revisions&rvprop=content'
dataformat = 'format=json'
formatversion = 'formatversion=2'
query_base = f'{baseurl}{action}&{content}&{dataformat}&{formatversion}&titles='

# %%
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
# %%
# for univ in univs:
# 	save_pages_from_df(data[univ], univ)

# %%
import re
wikilink_rx = re.compile(r'\[\[(.*?)\]\]')

def get_wikilinks(text):
	'''Return all wikilinks in a given text'''
	wikilinks = set()
	for l in wikilink_rx.finditer(text):
		wikilinks.add(l[1].split('|')[0])
	return wikilinks

def filter_hero_links(wikilinks, hero_wikilinks):
	'''Returns intersection between wikilinks and hero wikilinks'''
	return set(wikilinks).intersection(set(hero_wikilinks))

#%%
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
	for univ in univs:
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

# %%
# Build hero graph
hero_graph = nx.DiGraph()
for univ in univs:
	data[univ].apply(update_graph, args=(univ,), axis=1)

# %%
# Remove nodes without in/out edges
hero_graph.remove_nodes_from(list(nx.isolates(hero_graph)))
# %%
# Find GCC of hero graph
hero_gcc = hero_graph.subgraph(sorted(nx.weakly_connected_components(hero_graph), key=len, reverse=True)[0])
hero_graph = hero_gcc # to avoid later confusion

# %%
# EX 1a
# What is the number of nodes in the network?
# What is the number of links?

print('Number of nodes in GCC:', len(hero_gcc.nodes))
print('Number of links in GCC:', len(hero_gcc.edges))
# TODO: COMMENT ON THAT

# %%
# EX 1a
# What is the number of links connecting Marvel and DC? Explain in your own words what those links mean?

def count_cross_universe_links(node, graph):
	universe = node[1]['universe']
	s = 0
	for n in graph.neighbors(node[0]):
		if graph.nodes[n]['universe'] != universe:
			s += 1
	return s

# cul - cross universe links
total_cul = 0
for node in hero_gcc.nodes(data=True):
	total_cul += count_cross_universe_links(node, hero_gcc)

print('Number of cross-universe links', total_cul)
# TODO: COMMENT ON THAT

# %%
# EX 1a
# Who are top 5 most connected characters? (Report results for in-degrees and out-degrees). Comment on your findings. Is this what you would have expected.

# Top 5 'visited' nodes
top_in = sorted(hero_gcc.in_degree(), key=lambda x: x[1], reverse=True)[:5]
# Top 5 'visiting' nodes
top_out = sorted(hero_gcc.out_degree(), key=lambda x: x[1], reverse=True)[:5]

def report_heroes(heroes, dir):
	print(f'Top {len(heroes)} {dir} connected heroes')
	print(f'{"Name:": <35}Links:')
	for h in heroes:
		print(f'{h[0]: <35}{h[1]}')

report_heroes(top_in, 'in')
print()
report_heroes(top_out, 'out')
# TODO: COMMENT ON THAT

# %%
# EX 1a
# Plot the in and out-degree distributions.
# 	Explain what you observe?
# 	Explain why the in-degree distribution is different from the out-degree distribution?

gcc_in_degrees = [degree for _, degree in hero_gcc.in_degree()]
gcc_out_degrees = [degree for _, degree in hero_gcc.out_degree()]

def barchart_distributions(data, title, caption, subplot=111):
	plt.subplot(subplot)
	vector = list(range(np.min(data), np.max(data) + 2))
	graph_values, graph_bins = np.histogram(data, bins=vector)
	plt.bar(graph_bins[:-1], graph_values, width=.8)
	plt.title(title)
	plt.xlabel('Node degree')
	plt.ylabel('Count')
	plt.grid()
	plt.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=12)

plt.figure(figsize=(14, 6))
caption = 'Figure 1. Histograms showing distributions of node degrees in\ngraph visualizing connections between Wikipedia pages of Marvel and DC heroes.'
title = 'In-degree distribution in GCC of Marvel & DC hero graph'
barchart_distributions(gcc_in_degrees, title, caption, 121)
title = 'Out-degree distribution in GCC of Marvel & DC hero graph'
barchart_distributions(gcc_out_degrees, title, caption, 122)
plt.show()
# TODO: COMMENT ON THAT

# %%
# EX 1a
# Compare the degree distribution to a random network with the same number of nodes and probability of connection $p$. Comment or your results.
size, p = hero_gcc.order(), 0.002
random_network = nx.erdos_renyi_graph(size, p)
random_degrees = [degree for _, degree in random_network.degree()]
gcc_degrees = [degree for _, degree in hero_gcc.degree()]
plt.figure(figsize=(14, 6))
caption = 'Figure 2. Comparison between node degree distributions of random network and hero network.\n'
title = f'Degree distribution in Erdös-Renyi network\nN={size}, p={p}'
barchart_distributions(random_degrees, title, caption, 121)
title = f'Degree distribution in GCC of Marvel & DC hero graph'
barchart_distributions(gcc_degrees, title, caption, 122)
plt.show()
# TODO: Comment on that

# %%
# EX 1b
# Create a nice visualization of the total network.
# 	Color nodes according to universe
# 	Scale node-size according to degree
# 	Whatever else you feel like.
# 	If you can get it to work, get node positions based on the Force Atlas 2 algorithm

def get_node_color_map(graph):
	'''Returns color map taking universe into account'''
	color_map = []
	for node in graph.nodes(data=True):
		color_map.append('red' if node[1]['universe'] == 'marvel' else 'black')
	return color_map

def get_node_size_map(graph):
	'''Returns size map taking node degree into account'''
	degrees = dict(graph.degree)
	return [v for v in degrees.values()]

def get_edge_color(graph, n1 ,n2):
	univ1, univ2 = graph.nodes[n1]['universe'], graph.nodes[n2]['universe']
	if univ1 == univ2 == 'marvel':
		return 'blue'
	if univ1 == univ2 == 'dc':
		return 'yellow'
	return 'green'

def get_edge_color_map(graph):
	color_map = []
	for n1, n2 in graph.edges:
		color_map.append(get_edge_color(graph, n1, n2))
	return color_map

# %%
hero_undir = hero_gcc.to_undirected()
forceatlas2 = ForceAtlas2()
plt.figure(figsize=(14,8))
positions = forceatlas2.forceatlas2_networkx_layout(hero_undir, pos=None, iterations=500)
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
plt.show()
# %%
