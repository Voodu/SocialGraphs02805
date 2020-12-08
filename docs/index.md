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
2. Determining sentiment of the books and movies over time
    - TODO
    - TODO

## Graph analysis

First part concerned mainly exploratory analysis of the network and getting to know, what further can be done with the network. We analyzed basic properties, ex. node/edge count or distributions.

### Procedure

We performed analysis on several graph combinations:

-   books graph
-   movies graph
-   combined graph

For every situation we:

-   checked node & edge count
-   checked node degree distribution
-   determined most connected nodes
-   plotted the network

Further, we examined the communities exisiting in the network. #TODO

That analysis gave us good understanding of the network, connections between nodes, and its overall shape.

### Results

#### Basic analysis

-   Node/Edge counts

    | network  | node count | edge count |
    | -------- | ---------- | ---------- |
    | books    | 129        | 1871       |
    | movies   | 62         | 770        |
    | combined | 155        | 2136       |

-   Node degree distributions:

    ![Node degree distributions](images/nodes_distr.png 'Books community graph')

-   Top characters
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

-   Race-colored graphs

    **Books**
    ![Books race graph](images/basic_vis4.png 'Books race graph')

    **Movies**
    ![Movies race graph](images/basic_vis5.png 'Movies race graph')

    **Combined**
    ![Combined race graph](images/basic_vis5.png 'Combined race graph')

#### Community distribution

![Distribution of communities](images/distr.png 'Distribution of communities')

#### Community graphs

##### Books community graph

![Books community graph](images/graph_comm_books.png 'Books community graph')

##### Movies community graph

![Movie community graph](images/graph_comm_movie.png 'Movies community graph')

##### Combined community graph

![Combined community graph](images/graph_comm_comb.png 'Combined community graph')

#### TF-IDF wordclouds

![Wordclouds](images/wordclouds.jpg 'Wordclouds')

### Comments

Mollit esse ipsum sunt consequat officia quis et duis dolore. Adipisicing in id ea deserunt occaecat velit consectetur Lorem. Quis esse magna voluptate mollit. Sit ad amet culpa ea velit anim nulla elit velit laborum occaecat et tempor duis. Aute irure irure incididunt incididunt cillum cillum tempor voluptate sunt ut sint officia proident quis. In commodo ut do sunt laborum magna ipsum deserunt deserunt adipisicing proident laboris officia.

## Sentiment analysis

Id fugiat magna consectetur dolore velit elit velit aute ut fugiat cupidatat proident irure. Minim exercitation mollit ipsum culpa dolor id commodo. Nisi sit ea minim est minim. Do anim fugiat veniam ad incididunt do qui laboris sunt. Cupidatat ipsum occaecat sunt ea dolore labore irure cillum est veniam veniam. Ipsum consequat velit labore enim tempor reprehenderit velit. Ea nisi sunt laborum ea reprehenderit in laboris enim magna exercitation commodo sunt magna.

### Data

Dolore tempor non magna exercitation esse qui velit elit elit deserunt cillum elit qui. Officia consequat ex Lorem sunt exercitation adipisicing deserunt quis aute. Cupidatat magna nostrud fugiat nisi laborum reprehenderit adipisicing ex. Reprehenderit eiusmod amet aliquip et laborum irure deserunt officia veniam sint laboris. Ad elit cillum nostrud irure ea ad sit officia occaecat in adipisicing veniam officia. Et amet velit commodo minim reprehenderit Lorem et eu voluptate culpa occaecat veniam.

### Procedure

Ut occaecat qui enim officia irure minim. Fugiat sit magna eu Lorem laborum laboris. Aliquip eu qui consequat do magna sit eiusmod excepteur nulla nisi esse. Culpa cillum quis duis do commodo id cillum eu qui veniam laboris.

### Results

Aliquip id nisi esse nisi nisi do nisi nulla fugiat cillum et. Adipisicing dolor velit aliqua quis dolor Lorem fugiat cillum tempor incididunt. Irure ea sunt quis culpa laboris incididunt veniam reprehenderit laborum minim sint. Veniam culpa amet tempor occaecat nostrud do nostrud ut ullamco amet deserunt. In sit Lorem laboris qui duis deserunt. Deserunt ea adipisicing voluptate voluptate sit exercitation aute adipisicing pariatur occaecat est pariatur.

#### Sentiment over time

![Sentiment over time](images/plot_sentiment.png 'Sentiment over time')

#### Sentiment for communities

##### Books communities sentiment

|     | community | mean sentiment     |
| --- | --------- | ------------------ |
| 2   | frodo     | 5.490858970266937  |
| 0   | aragorn   | 5.485192309208288  |
| 1   | denethor  | 5.478623393959455  |
| 4   | saruman   | 5.475947778166416  |
| 3   | horn      | 5.3939784227845555 |

##### Movies communities sentiment

|     | community | mean sentiment    |
| --- | --------- | ----------------- |
| 1   | frodo     | 5.398292395031444 |
| 3   | gimli     | 5.392005576208235 |
| 0   | aragorn   | 5.388042271911186 |
| 4   | saruman   | 5.38325281640115  |
| 2   | gandalf   | 5.371323242476146 |

### Comments

Veniam aute incididunt velit amet occaecat commodo. Veniam occaecat voluptate aliqua ipsum velit labore consectetur laborum. Consequat ullamco fugiat incididunt dolor elit. Officia cupidatat cupidatat dolore ut. Sit enim velit pariatur adipisicing labore amet exercitation. Labore culpa velit deserunt qui elit mollit ex eiusmod.

## Closing remarks

Reprehenderit exercitation laboris cillum sit. Sunt do reprehenderit ut Lorem dolor. Aliquip duis proident sit magna dolor eiusmod nulla esse ut ut ut magna. Labore nisi irure pariatur labore ut qui commodo excepteur non ad. Dolore est elit enim laboris laboris cillum minim duis sit culpa laborum adipisicing. Eu occaecat et commodo Lorem laboris commodo laboris ea velit.

### Downloads & sources

[Jupyter notebook](https://nbviewer.jupyter.org/github/Voodu/SocialGraphs02805/blob/main/Project/main.ipynb)

[Dataset (list with names)](https://github.com/Voodu/SocialGraphs02805/blob/main/Project/data/characters.csv)

| Part | Movie script                                                                             | Book                                                                          |
| ---- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 1    | [Link](https://www.imsdb.com/scripts/Lord-of-the-Rings-Fellowship-of-the-Ring,-The.html) | [Link](http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_1__en.htm) |
| 2    | [Link](https://www.imsdb.com/scripts/Lord-of-the-Rings-The-Two-Towers.html)              | [Link](http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_2__en.htm) |
| 3    | [Link](https://www.imsdb.com/scripts/Lord-of-the-Rings-Return-of-the-King.html)          | [Link](http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_3__en.htm) |
