# Lord of the Rings - analysis

## General introduction

The project is about analyzing Lord of the Rings characters. By taking books and movie scripts we would like to understand connections between characters and check how does the sentiment in them change over time

### Dataset

Analysis will be performed over all 300-word chunks of text from book/script. Movie scripts are taken from https://www.imsdb.com/ and books are taken from http://ae-lib.org.ua/. Character information and names list are taken from [LotR wiki](https://lotr.fandom.com/wiki). It was chosen over normal wikipedia, because it has more comprehensive list of characters and their descriptions are structured very similarly (ex. it is easier to scrap race of every character).

Total number of characters is around 170, number of nodes in the biggest graph is 155 and there are over 2100 edges. There over 300.000 tokenized words in the whole dataset (books and movies). Character list is enriched with their race information taken from the wiki.

### Main goals

There are two main goals of the project:

1. Understanding, how the characters are connected:
    - Are there any strict communities or hubs?
    - Which characters are the most important?
    - Is there anything interesting about races?
2. Determining communities and sentiment of the books and movies over time
    - How do communities look like in books and movies?
    - Does TF-IDF wordclouds calculated using text chunks resemble the feelings of readers/viewers?
    - Does the sentiment align between books and movies?

## Graph analysis

First part concerned mainly exploratory analysis of the network and getting to know, what further can be done with the network. We analyzed basic properties, ex. node/edge count or distributions.

### Procedure

We performed analysis on several graph combinations:

-   books graph,
-   movies graph,
-   combined graph.

For every situation we:

-   checked node & edge count,
-   checked node degree distribution,
-   determined most connected nodes,
-   plotted the network.

That analysis gave us good understanding of the network, connections between nodes, and its overall shape.

### Results

#### Basic analysis
First of all, there's clear difference between number of nodes in books graph and movies graph. The former one has 149 nodes, while the latter one has only 62 of them. Obviously, it was easier for Tolkien to write about some character (or at least mention them) than for Jackson to find nearly 150 cast members (not including background actors). This leads to the conclusion that book is over 2 times more rich in content in terms of number of characters.

It is also interesting that combined graph contains 155 nodes, so 6 more than books. It means that some characters were created by movie director, even though they did not appear in the original book.

Another conclusion from the number of nodes and edges is that one has to read the books and watch the movies to have some basic knowledge about every character in the Lord of the Rings universe.

##### Node/Edge counts

| network  | node count | edge count |
| -------- | ---------- | ---------- |
| books    | 129        | 1871       |
| movies   | 62         | 770        |
| combined | 155        | 2136       |

##### Node degree distributions:

In general, one can think that the degree distributions follow the power-law, but it is not exactly the case here. For network to be scale-free, there should be big number of nodes with the lowest degree (i.e. 1) and it should exponentially decrease with the degree. Here, on every plot the highest count is around degree 10. It decreases later, but it does not happen smoothly. One can say, at most, that those networks are 'scale-free-like'. It makes perfect sense, though, as they are not typical social networks or semantic networks. They are artificially build using predefined chunk size. They may be considered as semi-social networks, as they are build using character interactions.

It is also quite interesting that nodes in movie network have lower degree in general - they vary between 1 and around 60, while in books they go from 1 to around 120. This may be caused by the different nature of media - in a book a character can think about someone else who is not currently present with them. In a movie, there are no character thoughts and characters interact only with those present in the scene with them. Therefore, there is less 'mixing' of the characters, as they are contained in their scenes. Moreover, there are just more characters mentioned in the books, as seen in the node counts.

![Node degree distributions](images/nodes_distr.png 'Books community graph')

##### Top characters

The most connected nodes in all the graphs are partially similar, but not as much as one may assume. All of them include some of the most important characters like four hobbits, Gandalf or Aragorn. Books are more focused on hobbits, i.e. Frodo, Sam, Merry, and Pippin, as they occupy top positions in the ranking, while movies are more focused on Aragorn, Legolas, Gimli, Gandalf and Frodo - the team one usually sees on the movie posters. The biggest surprise is very high degree of Bilbo in books. It may be caused by the fact, the he is uncle of Frodo who mentions him quite often in the book. In movie there is no way to show character's thoughts, so Bilbo is not as highly ranked as in the book.


<table>
<tr><th>Books</th><th>Movies</th><th>Combined</th></tr><tr><td>

| Name    | Degree |
| ------- | ------ |
| Gandalf | 108    |
| Frodo   | 103    |
| Pippin  | 103    |
| Merry   | 99     |
| Horn    | 99     |

</td><td>

| Name    | Degree |
| ------- | ------ |
| Aragorn | 59     |
| Legolas | 57     |
| Gimli   | 56     |
| Gandalf | 52     |
| Frodo   | 51     |

</td><td>

| Name    | Degree |
| ------- | ------ |
| Gandalf | 120    |
| Pippin  | 111    |
| Frodo   | 109    |
| Merry   | 106    |
| Horn    | 102    |

</td></tr></table>

#### Race-colored graphs
The graphs confirm observations from the previous points - as books have more nodes and more connections between them, the graph looks way more dense than the movie one.

Also, thanks to the coloring of nodes, it can be observed that, although there are many creature races in the book and movies, vast majority of the characters in the story are men or hobbits.

**Books**
![Books race graph](images/basic_vis4.png 'Books race graph')

**Movies**
![Movies race graph](images/basic_vis5.png 'Movies race graph')

**Combined**
![Combined race graph](images/basic_vis5.png 'Combined race graph')

## Community analysis
### Data
The data to accomplish this exercise comes mainly from the previous exercise.

### Procedure
We examined the communities existing in the network:

-  finding communities and providing basic statistics about them,
-  plotting graphs depending on communities to check how characters are connected,
-  creating TF-IDF for communities and text chunks for each community.

#### Community distribution
The sizes are similar between the sources. The interesting fact is that combined communities have greater number of nodes in each community and it contains one more community than book and movie sources. Movie communities are the smallest because due to the much smaller amount of words, fewer characters were introduced explicitly. It seems that book communities fill the gap in the combined option.

![Distribution of communities](images/distr.png 'Distribution of communities')

#### Community graphs
After plotting the graphs where nodes are coloured by the community, it can be seen that communities not only are bounded by the race of the characters but additionally by their relationship and the area (the chapter or section of occurence) in the books and movies. This example could be presented by the connections for Bilbo. On the movie graph, this relation is much less significant than in the books which places Bilbo as more significant character in books than in movies. Also in books Bilbo has got much more connections inside of his community than in movies. Moreover, Merry and Pippin are in different communities in books but they are highly significant. On the other hand, in movies they are presented almost always together thus they are in one community. Combined graph seem to have combined properties of those facts mentioned above.

##### Books community graph

![Books community graph](images/graph_comm_books.png 'Books community graph')

##### Movies community graph

![Movie community graph](images/graph_comm_movie.png 'Movies community graph')

##### Combined community graph

![Combined community graph](images/graph_comm_comb.png 'Combined community graph')

#### TF-IDF wordclouds
For presentation purposes to show communities in one source, only the book communities were chosen to present wordclouds. Not every wordcloud seems to present significant results, however some of them could be discussed.
* "Théoden" – one can see words like Rohan, king, horse – those words are associated with this character.
* "Elrond" – the Fellowship of the Ring is present here.
* "Frodo" – words seem to be much bigger and there are associations like Sam, Bilbo, dark, eye, hobbit.

![Wordclouds](images/wordclouds.png 'Wordclouds')

### Comments

The communities seem to be correct. Further investigation could be required to answer the questions such as "Why Merry is inside of the orcs and ents community in the books?". Moreover, it turned out that chosen method for creating wordclouds (selecting chunks where the community member name exists and adding the chunk to community text container) provides fine results.

## Sentiment analysis

### Data

Two groups of texts were used in this exercise. Firstly, the source books and movies scripts were split in chunks to calculate rolling sentiment over time. Secondly, during the creation of wordclouds, the community texts were created for which the sentiment will be calculated.

### Procedure

To calculate the sentiment we:

- compared sentiment over time in every part of book and movie,
- calculated sentiment in each community.

### Results

#### Sentiment over time
Below, the comparison for each part of the trilogy in both sources was presented. The movie sentiment fluctuates much more than the book one - probably because of the amount of words in each source. Nonetheless, the lines are relatively aligned, especially, especially during the climaxes at the end of first and third part of the trilogy. What is worth mentioning is the fact that there is no guarantee that fact in books and movies are presented at the same order. Moreover, the viewers feel the sentiment from the movie frames mainly, thus not every moment in the films might be properly interpreted by the average sentiment. The sudden drops in sentiment at the end are caused by small size of the last chunk - chunks are not evenly sized.

![Sentiment over time](images/plot_sentiment.png 'Sentiment over time')

#### Sentiment for communities
The happiest community throughout the books is "Merry", probably because of the fact that this is small community (see the graph) and the Merry is mostly connected character there – he is very positive character. The saddest community is represented by "Horn" – the orc surrounded by others of the same race which is an obvious fact – they are representing the evil in the world of Middle-earth.

As far as movies are concerned, the situation is very similar to the books. What is different here, the saddest community is represented by Denethor – the Steward of Gondor who wanted to burn his son alive and his community also contains negative characters like orcs. In this community there is also Frodo – probably walking with ring has surrounded him with a lot of negativity.

<table><tr><th>Books</th><th>Movies</th></tr><tr><td>

| community | mean sentiment     |
| --------- | ------------------ |
| merry     |     5.521230       |
| théoden   |     5.481627       |
| elrond    |     5.477176       |
| frodo     |     5.474550       |
| pippin    |     5.466320       |
| horn      |     5.373511       |
</td><td>

| community | mean sentiment    |
| --------- | ----------------- |
| merry     | 5.493679          |
| gandalf   | 5.406756          |
| théoden   | 5.382372          |
| gamling   | 5.360385          |
| denethor  | 5.348106          |
| &nbsp;    | &nbsp;            |
</td></tr></table>


## Closing remarks

Our results expose the most important differences and indicate biggest similarities between books and movies. While most of them are not surprising, some of them are thought provoking, ex. how big is the difference between number of characters in the books and movies.

To provide more interesting network analysis, more manual work with extended domain knowledge is required. Unfortunately, wiki is not complete, i.e. there are very few common attributes as race for every character. If there were additional information (ex. conflict side), more interesting insights would be possible to derive. The above analysis still enabled to understand the differences between book and movies, but definitely can be extended by providing additional data.

### Downloads & sources

[Jupyter notebook](https://nbviewer.jupyter.org/github/Voodu/SocialGraphs02805/blob/main/Project/main.ipynb)

[Dataset (list with names)](https://github.com/Voodu/SocialGraphs02805/blob/main/Project/data/characters.csv)

| Part | Movie script                                                                             | Book                                                                          |
| ---- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 1    | [Link](https://www.imsdb.com/scripts/Lord-of-the-Rings-Fellowship-of-the-Ring,-The.html) | [Link](http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_1__en.htm) |
| 2    | [Link](https://www.imsdb.com/scripts/Lord-of-the-Rings-The-Two-Towers.html)              | [Link](http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_2__en.htm) |
| 3    | [Link](https://www.imsdb.com/scripts/Lord-of-the-Rings-Return-of-the-King.html)          | [Link](http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_3__en.htm) |
