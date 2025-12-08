############################################
# Authors: Mostafa Hosseini & Foad Bagheri #
#               12/08/2025                 #
############################################

import networkx as nx
from sklearn.cluster import SpectralClustering

def sanitize(x):
    return x.strip(',;\n ()')

def parser(file_):
    with open(file_, 'r') as bench_file:
        input_nodes = []
        output_nodes = []
        gates = []
        wires = []

        for line in bench_file:
            line = line.strip()
            if line.startswith("INPUT"):
                node = sanitize(line.lstrip("INPUT"))
                input_nodes.append(node)
            elif line.startswith("OUTPUT"):
                node = sanitize(line.lstrip("OUTPUT"))
                output_nodes.append(node)
            elif "=" in line:
                output_port, gate_definition = line.split("=")
                output_port = sanitize(output_port)
                gate_type, ports = gate_definition.strip().split("(")
                ports = [sanitize(x) for x in ports.split(",")]
                gate_name = output_port
                gates.append(gate_name)

                existing_wire = next((w for w in wires if w[0] == output_port), None)
                if existing_wire:
                    existing_wire[1] = gate_name
                else:
                    wires.append([output_port, gate_name, []])

                for i in ports:
                    existing_wire = next((w for w in wires if w[0] == i), None)
                    if existing_wire:
                        existing_wire[2].append(gate_name)
                    else:
                        wires.append([i, i if i in input_nodes else ' ', [gate_name]])

    return input_nodes, output_nodes, gates, wires


def grapher(in_n, out_n, nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(in_n)
    G.add_nodes_from(out_n)
    G.add_nodes_from(nodes)

    for i in edges:
        for j in i[2]:
            G.add_edge(i[1], j, weight=6)
    return G


def cluster_inputs_by_rare_nodes(graph, input_nodes, Rare_nodes, num_clusters=3):
    # Filter Rare_nodes to ensure they exist in the graph (Validation)
    valid_rare_nodes = [n for n in Rare_nodes if n in graph.nodes]

    # Create subgraph including inputs and valid rare nodes
    sub_nodes = set(input_nodes).union(valid_rare_nodes)
    subgraph = graph.subgraph(sub_nodes)

    # Create adjacency matrix
    adj_matrix = nx.to_numpy_array(subgraph)
    # Make it symmetric
    adj_matrix = (adj_matrix + adj_matrix.T) / 2

    # Perform Spectral Clustering
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed').fit(adj_matrix)
    labels = clustering.labels_

    # Map results back to inputs
    clustered_inputs = {}
    subgraph_nodes = list(subgraph.nodes)

    for idx, label in enumerate(labels):
        node = subgraph_nodes[idx]
        # Only care if the node is an input
        if node in input_nodes:
            if label not in clustered_inputs:
                clustered_inputs[label] = []
            # Append the NODE NAME (String), not int, to match Parser
            clustered_inputs[label].append(node)

    return clustered_inputs