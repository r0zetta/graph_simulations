from graph import *

conf = {"cores": [50, 100, 50],
        "core_core_connections": [[0, 1, 5], [1, 2, 10]],
        "core_core_connection_type": [["direct"], ["indirect"]],
        "extra_nodes_random": [20, 0, 30],
        "extra_nodes_popularity": [50, 100, 10],
        "popularity_cutoff": [1.0, 0.2, -0.2],
        "extra_edges_second_neighbour": [25, 30, 10],
        "extra_edges_random": [5, 0, 20]}

g = Graph(config=conf)
g.print_basic_stats()
g.print_community_stats()

# Play with graph construction here
g = Graph(num_nodes=300,
          num_cores=1,
          core_connectivity=1.0,
          add_nodes_random=0.5,
          add_nodes_popularity=1.5,
          popularity_cutoff=0.5,
          connect_cores_directly=0.0,
          connect_second_neighbours=1.0,
          move_nodes_second_neighbour=0.5,
          connect_random=0.0)
g.print_basic_stats()
g.print_community_stats()
g.write_gexf("graph_2.gexf")

