# graph_simulations
Synthesising graphs and simulating things

This repository contains tools for creating, manipulating, and visualizing node-edge network graph representations. It also contains some code that puts those tools to use. The tools are described in the following sections.

# graph.py

graph.py is a python program for creating node-edge graphs from scratch. The tool is highly configurable, allowing for the creation of many distinct node-edge graph phenotypes. The tool requires you have the following python libraries installed:

**networkx** (https://networkx.org/)
`pip install networkx`

**louvain community detection** (https://github.com/taynaud/python-louvain)
`pip install python-louvain`

The following code creates a graph using default settings and then prints some statistics about it.
```
from graph import *

g = Graph()
g.print_basic_stats()
g.print_community_stats()
```

The point of the graph.py tool is to allow researchers to create node-edge graphs with interesting properties. The properties of the generated graphs can be studied, and they can also be utilized in, for instance, simulations (more on that later). The Graph() initialization routine allows for the following parameters:

**num_nodes** (default:1000) is a value that is used in graph generation. It does not specifically determine the final number of nodes in a generated graph. However, the larger the **num_nodes** value, the larger the graph.

**num_cores** (default: 1) in the initial phase of graph generation, a number of cores are created in the following way - a set of nodes (roughly equal to num_nodes/20 * num_cores) is created for each core. These nodes are then connected to one another based on the **intra_core_connectivity** variable. Cores are then connected together (based on variable described below), and additional nodes are finally added to the entire graph. The minimum value for **num_cores** is 1.

**intra_core_connectivity** (default: 0.3) defines the density of connections inside the initially created cores. Higher values add more edges during initial core formation.

**core_connectivity** (default: 0.7) defines the density of connectivity between initially formed cores. The mean number of nodes across all created cores is multiplied by this coefficient to determine the number of connections to be made between cores. Each time two cores are connected, the cores to be connected are determined at random. Hence, for larger values of **num_cores**, larger values of **core_connectivity** may be required.

**add_nodes_random** (default: 0.4) After cores have been created and connected together, additional nodes are added to the graph. In one case, new nodes are created and connected to other nodes at random. This value is multiplied by **num_nodes** to determine how many nodes are created in this way.

**add_nodes_popularity** (default: 1.4) After cores have been created and connected together, additional nodes are added to the graph. In one case, new nodes are created and connected to other nodes such that nodes with higher numbers of connected nodes are more likely to further receive new nodes. This value is multiplied by **num_nodes** to determine how many nodes are created in this way.

**popularity_cutoff** (default: 1.0) When choosing a node such that existing connections weight the node more likely to be chosen, all nodes are still considered in the final categorical distribution. To limit the number of nodes available for selection, one may set **popularity_cutoff** to a value between 0 and 1. A lower value will select a smaller portion of the most connected nodes.

**connect_cores_directly** (default: 0.2) defines the chance that nodes belonging to cores are connected directly to other cores, or via intermediate nodes. Setting this value to 0 will guarantee that cores are always connected via intermediate nodes. Setting the value to 1 ensures cores are always connected directly.

**connect_second_neighbours** (default: 1.5) after random nodes are added to the graph, some additional edges are created between existing nodes. Some connections are intentionally formed between a node and its second neighbour. The number of connections made this way is determined by multiplying **num_nodes** with **connect_second_neighbours**.

**connect_random** (default: 0.4) after random nodes are added to the graph, some additional edges are created between existing nodes. Some connections are intentionally formed between a randomly selected pair of nodes. The number of connections made this way is determined by multiplying **num_nodes** with **connect_random**.

