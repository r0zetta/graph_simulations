from graph import *
import os, time


def print_voting_intention(voting_intention):
    vote_blue = 0
    vote_red = 0
    abstain = 0
    for name, vi in voting_intention.items():
        if vi > voting_threshold:
            vote_blue += 1
        elif vi < -1*voting_threshold:
            vote_red += 1
        else:
            abstain += 1
    blue_per = vote_blue / len(voting_intention)
    red_per = vote_red / len(voting_intention)
    turnout = (vote_red + vote_blue)/len(voting_intention)
    msg = "Red: " + str(vote_red)
    msg += "  Blue: " + str(vote_blue)
    msg += "  Abstain: " + str(abstain)
    msg += "  Turnout: " + "%.2f"%(turnout*100)
    print(msg)
    return red_per, blue_per, turnout

def print_community_voting_intention(graph, voting_intention):
    blue_seats = 0
    red_seats = 0
    for cluster_index, cluster_nodes in graph.clusters.items():
        vi = 0
        vi = np.mean([voting_intention[x] for x in cluster_nodes])
        voted_blue = 0
        voted_red = 0
        abstained = 0
        for x in cluster_nodes:
            if voting_intention[x] > voting_threshold:
                voted_blue += 1
            elif voting_intention[x] < -1 *voting_threshold:
                voted_red += 1
            else:
                abstained += 1
        adit = ""
        winner = "tie"
        if voted_blue > voted_red:
            winner = "Blue"
            blue_seats += 1
            if (voted_blue/(voted_blue + voted_red)) > 0.65:
                adit = " (safe)"
        elif voted_red > voted_blue:
            winner = "Red"
            red_seats += 1
            if (voted_red/(voted_blue + voted_red)) > 0.65:
                adit = " (safe)"
        sign = ""
        msg = "Cluster: " + "%02d"%cluster_index
        msg += " (" + "%03d"%len(cluster_nodes) + ")"
        msg += "  voted blue: " + "%02d"%voted_blue
        msg += "  voted red: " + "%02d"%voted_red
        msg += "  abstained: " + "%02d"%abstained
        msg += "  seat to: " + winner + adit
        print(msg)
    print("Red seats: " + str(red_seats) + "  Blue seats: " + str(blue_seats))
    red_per = red_seats/len(graph.clusters)
    blue_per = blue_seats/len(graph.clusters)
    return red_per, blue_per


def alter_voting_intention(voting_intention, magnitude):
    ret = {}
    sav = (scandal_min + scandal_max) / 2
    mod = random.choice([-1,1]) * sav * scandal_effect
    mag = magnitude + mod
    rt = resistance_threshold*0.99
    affected = int(len(voting_intention)*scandal_population_affected)
    to_alter = random.sample(range(len(voting_intention)), affected)
    for name, vi in voting_intention.items():
        if name not in to_alter:
            ret[name] = vi
            continue
        if solid_voters == True and vi > resistance_threshold:
            ret[name] = vi
        elif solid_voters == True and vi < -1 * resistance_threshold:
            ret[name] = vi
        else:
            ret[name] = max(-1*rt, min(1*rt, vi + mag))
    return ret

def propagate_voting_intention(graph, voting_intention):
    ret = {}
    rt = resistance_threshold*0.99
    for nodeid, vi in voting_intention.items():
        if solid_voters == True:
            if vi > resistance_threshold or vi < -1 * resistance_threshold:
                ret[nodeid] = vi
                continue
        neighbours = graph.get_incoming_weights(nodeid)
        outgoing = graph.get_outgoing_weights(nodeid)
        for item in outgoing:
            n, w = item
            neighbours.append([n,1])
        if len(neighbours) < 1:
            ret[nodeid] = vi
            continue
        #ns = random.sample(neighbours, max(1, int(len(neighbours)*0.5)))
        ns = neighbours
        nodes = [x[0] for x in ns]
        weights = [x[1] for x in ns]
        mw = max(weights)
        if len(nodes) > 4:
            nodes = random.sample(nodes, random.randint(4, len(nodes)))
        vil = [voting_intention[x] * (weights[i]/mw) for i, x in enumerate(nodes)]
        mnvi = np.median(vil)
        change = (vi - mnvi) * neighbour_influence
        change += random.uniform(-1*waver, waver)
        ret[nodeid] = max(-1*rt, min(1*rt, vi + change))
    return ret

num_nodes=1000
num_cores=1
core_connectivity=0.7
add_nodes_random=0.4
add_nodes_popularity=1.4
popularity_cutoff=0.8
connect_cores_directly=0.2
connect_second_neighbours=1.5
move_nodes_second_neighbour=0.5
connect_random=0.4

solid_voters = False
voting_threshold = 0.7
resistance_threshold = 0.95
neighbour_influence = 0.2
scandal_min = 0.2
scandal_max = 0.4
scandal_effect = 0.75
scandal_max_chance = 0.75
scandal_population_affected=0.3
waver = 0.5
move_nodes = int(num_nodes * 0.005)
move_every = 25
change_weights = int(num_nodes * 0.01)
change_every = 50
num_steps = 10000

save_every=500

# Make population interaction graph
graph = Graph(num_nodes=num_nodes,
              num_cores=num_cores,
              core_connectivity=core_connectivity,
              add_nodes_random=add_nodes_random,
              add_nodes_popularity=add_nodes_popularity,
              connect_cores_directly=connect_cores_directly,
              connect_second_neighbours=connect_second_neighbours,
              move_nodes_second_neighbour=move_nodes_second_neighbour,
              connect_random=connect_random)
graph.save_graph_csv("graph.csv")
graph.print_community_stats()
g_nodes = len(graph.nodeids)

# Seed initial voting intention to a few influencers in each community
voting_intention = Counter()
num_influencers = int(g_nodes*0.1)
blue_inf = 0
red_inf = 0
turn = "blue"
for index in range(g_nodes):
    voting_intention[index] = 0.0
for cluster_index, cluster_nodes in graph.clusters.items():
    cluster_indegree = Counter()
    for x in cluster_nodes:
        cluster_indegree[x] = graph.in_degree[x]
    c_inf = int((len(cluster_nodes)/g_nodes) * num_influencers)
    influencers = [x for x, c in cluster_indegree.most_common(c_inf)]
    intention = 0.0
    if turn == "blue":
        for i in influencers:
            intention = random.uniform(0.93, 1.0)
            voting_intention[i] = intention
        turn = "red"
    else:
        for i in influencers:
            intention = random.uniform(-1.0, -0.93)
            voting_intention[i] = intention
        turn = "blue"

# Propagate voting intention out to all other nodes
for _ in range(5):
    voting_intention = propagate_voting_intention(graph, voting_intention)

# Add voting intention values to be used by gephi graph output
graph.set_node_desc("voting_intention", "float", voting_intention)

# Modify how node_attr works, so it can be easily updated on each pass
# Add initial voting intention
# Add age
# Add local demographics
# Add media influence

# Classify different graphs based on initial starting values
# Print them visually
# "Evolve" graphs using different node move strategies

graph.write_gexf("graph.gexf")

stats = {}
for label in ["turnout",
              "red_seat_per",
              "blue_seat_per",
              "red_vote_per",
              "blue_vote_per",
              "blue_scandal",
              "red_scandal"]:
    stats[label] = []
last_scandal_step = 0
scandal_magnitude = 0
scandal_total = 0
scandal_side = 0
scandal_duration = 0.065
next_scandal = int(0.5/scandal_duration)*8
use_scandal = True
for n in range(num_steps):
    if n == next_scandal:
        scandal_side = random.choice([-1,0,0,0,1])
        scandal_duration = random.uniform(0.04, 0.07)
        next_scandal = n + int(1/scandal_duration)*4
    scandal_chance = max(0, abs(np.sin(n*scandal_duration))-(1.0-scandal_max_chance))
    os.system('clear')
    print("Nodes: " + str(g_nodes) + " Step: " + str(n))
    print("Scandal chance: " + "%.2f"%scandal_chance)
    print("Scandal side: " + "%.2f"%scandal_side)
    print("Scandal length: " + str(int(1/scandal_duration)*4))
    print("Next scandal: " + str(next_scandal))
    print("Last scandal: " + "%.2f"%scandal_magnitude + " on step: " + str(last_scandal_step))
    print("Total scandal: " + "%.2f"%scandal_total)
    red_vote_per, blue_vote_per, turnout = print_voting_intention(voting_intention)
    stats['red_vote_per'].append(red_vote_per)
    stats['blue_vote_per'].append(blue_vote_per)
    stats['turnout'].append(turnout)
    red_seat_per, blue_seat_per = print_community_voting_intention(graph, voting_intention)
    stats['red_seat_per'].append(red_seat_per)
    stats['blue_seat_per'].append(blue_seat_per)
    if n > 0 and n % save_every == 0:
        with open("stats.json", "w") as f:
            f.write(json.dumps(stats))
    if use_scandal == True and random.random() < scandal_chance:
        scandal_magnitude = random.uniform(scandal_min, scandal_max) + (0.08*scandal_chance)
        scandal_magnitude = scandal_side * scandal_magnitude
        if random.random() < 0.2:
            scandal_magnitude = scandal_magnitude * -1
        if scandal_magnitude != 0:
            last_scandal_step = n
        if scandal_magnitude < 0:
            stats['blue_scandal'].append(abs(scandal_magnitude)*0.2)
            stats['red_scandal'].append(0)
        elif scandal_magnitude > 0:
            stats['red_scandal'].append(abs(scandal_magnitude)*0.2)
            stats['blue_scandal'].append(0)
        voting_intention = alter_voting_intention(voting_intention, scandal_magnitude)
        scandal_total += scandal_magnitude
    else:
        stats['blue_scandal'].append(0)
        stats['red_scandal'].append(0)
    voting_intention = propagate_voting_intention(graph, voting_intention)
    if move_nodes > 0:
        if n % move_every == 0:
            to_move = random.sample(range(g_nodes), move_nodes)
            graph.move_nodes(to_move)
    if change_weights > 0:
        if n % change_every == 0:
            to_increase = random.sample(range(g_nodes), int(change_weights/2))
            graph.increase_weights(to_increase)
            to_decrease = random.sample(range(g_nodes), int(change_weights/2))
            graph.decrease_weights(to_decrease)
with open("stats.json", "w") as f:
    f.write(json.dumps(stats))
