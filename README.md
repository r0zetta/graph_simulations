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

**core_connectivity** (default: 0.7) defines the density of connectivity between initially formed cores. The mean number of nodes across all created cores is multiplied by this coefficient to determine the number of connections to be made between cores. Each time two cores are connected, the cores to be connected are determined at random. Hence, for larger values of **num_cores**, larger values of **core_connectivity** may be required. Note that this setting is ignored when **num_cores** is 1.

**add_nodes_random** (default: 0.4) After cores have been created and connected together, additional nodes are added to the graph. In one case, new nodes are created and connected to other nodes at random. This value is multiplied by **num_nodes** to determine how many nodes are created in this way.

**add_nodes_popularity** (default: 1.4) After cores have been created and connected together, additional nodes are added to the graph. In one case, new nodes are created and connected to other nodes such that nodes with higher numbers of connected nodes are more likely to further receive new nodes. This value is multiplied by **num_nodes** to determine how many nodes are created in this way.

**popularity_cutoff** (default: 1.0) When choosing a node such that existing connections weight the node more likely to be chosen, all nodes are still considered in the final categorical distribution. To limit the number of nodes available for selection, one may set **popularity_cutoff** to a value between 0 and 1. A lower value will select a smaller portion of the most connected nodes.

**connect_cores_directly** (default: 0.2) defines the chance that nodes belonging to cores are connected directly to other cores, or via intermediate nodes. Setting this value to 0 will guarantee that cores are always connected via intermediate nodes. Setting the value to 1 ensures cores are always connected directly. Note that this setting is ignored when **num_cores** is 1.

**connect_second_neighbours** (default: 1.5) after random nodes are added to the graph, some additional edges are created between existing nodes. Some connections are intentionally formed between a node and its second neighbour. The number of connections made this way is determined by multiplying **num_nodes** with **connect_second_neighbours**.

**connect_random** (default: 0.4) after random nodes are added to the graph, some additional edges are created between existing nodes. Some connections are intentionally formed between a randomly selected pair of nodes. The number of connections made this way is determined by multiplying **num_nodes** with **connect_random**.

The above description of graph.py's initialization options probably doesn't make sense. However, I will illustrate how the parameters work with a few examples.

## Example 1: default settings

Default settings are as follows:
```
g = Graph(num_nodes=1000,
          num_cores=1,
          intra_core_connectivity=0.3,
          core_connectivity=0.7,
          add_nodes_random=0.4,
          add_nodes_popularity=1.4,
          popularity_cutoff=1.0,
          connect_cores_directly=0.2,
          connect_second_neighbours=1.5,
          connect_random=0.4)
 ```
![generated_1](media/generated_1.png)

## Example 2: no random nodes or connections formed

In this config you will notice more "umbrella" clusters connected to nodes with high in-degree.
```
g = Graph(num_nodes=1000,
          num_cores=1,
          intra_core_connectivity=0.3,
          core_connectivity=0.3,
          add_nodes_random=0.0,
          add_nodes_popularity=1.0,
          popularity_cutoff=1.0,
          connect_cores_directly=0.2,
          connect_second_neighbours=1.0,
          connect_random=0.0)
```
![generated_2](media/generated_2.png)

# Example 3: classic two-core network

This is an example of how to create a loosely-connected two-core network. Such patterns are commonly seen when studying political content on social networks.
```
g = Graph(num_nodes=1000,
          num_cores=2,
          intra_core_connectivity=0.3,
          core_connectivity=0.2,
          add_nodes_random=0.0,
          add_nodes_popularity=1.0,
          popularity_cutoff=1.0,
          connect_cores_directly=0.0,
          connect_second_neighbours=1.0,
          connect_random=0.0)
```
![generated_3](media/generated_3.png)

# Example 4: loosely-connected three-core network

Such networks are rare in the real-world, but may be of interest to study.
```
g = Graph(num_nodes=1000,
          num_cores=3,
          intra_core_connectivity=0.3,
          core_connectivity=0.2,
          add_nodes_random=0.0,
          add_nodes_popularity=1.0,
          popularity_cutoff=0.5,
          connect_cores_directly=0.0,
          connect_second_neighbours=1.0,
          connect_random=0.0)
```
![generated_4](media/generated_4.png)

# Example 5: loosely connected disparate comunities

This is an example of what happens when **num_cores** and **core_connectivity** are set to very high values. This pattern is sometimes seen in follower-following interactions of botnet accounts on Twitter. Each separate cluster is highly connected and forms its own community. Such a graph might be interesting for simulation purposes.
```
g = Graph(num_nodes=1000,
          num_cores=6,
          intra_core_connectivity=0.1,
          core_connectivity=3.0,
          add_nodes_random=0.0,
          add_nodes_popularity=1.0,
          popularity_cutoff=0.5,
          connect_cores_directly=0.5,
          connect_second_neighbours=1.0,
          connect_random=0.0)
```
![generated_5](media/generated_5.png)

# Example 6: blobs

This configuration illustrates what happens when **add_nodes_popularity** is set to a very high value and **popularity_cutoff** is set to a very low value.
```
g = Graph(num_nodes=2000,
          num_cores=2,
          intra_core_connectivity=0.1,
          core_connectivity=0.5,
          add_nodes_random=0.1,
          add_nodes_popularity=3.0,
          popularity_cutoff=0.2,
          connect_cores_directly=0.1,
          connect_second_neighbours=0.5,
          connect_random=0.1)
```
![generated_6](media/generated_6.png)

# Example 7: jellyfish

This configuration also illustrates what happens when **add_nodes_popularity** is set to a very high value and **popularity_cutoff** is set to a very low value. However, in this example, **intra_core_connectivity** is also set high.
```
g = Graph(num_nodes=1000,
          num_cores=1,
          intra_core_connectivity=0.8,
          core_connectivity=0.5,
          add_nodes_random=0.1,
          add_nodes_popularity=3.0,
          popularity_cutoff=0.4,
          connect_cores_directly=0.1,
          connect_second_neighbours=0.3,
          connect_random=0.0)
```
![generated_7](media/generated_7.png)

# Example 8: binary system

This is a two-core example of the scenario where **add_nodes_popularity** is set to a very high value and **popularity_cutoff** is set to a very low value. In this example, **core_connectivity** and **connect_cores_directly** are both low values.
```
g = Graph(num_nodes=2000,
          num_cores=2,
          intra_core_connectivity=0.7,
          core_connectivity=0.2,
          add_nodes_random=0.5,
          add_nodes_popularity=2.0,
          popularity_cutoff=0.4,
          connect_cores_directly=0.05,
          connect_second_neighbours=0.3,
          connect_random=0.1)
```
![generated_8](media/generated_8.png)

