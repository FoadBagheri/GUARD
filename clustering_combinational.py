############################################
# Authors: Mostafa Hosseini & Foad Bagheri #
#               12/08/2025                 #
############################################

import networkx as nx
from sklearn.cluster import SpectralClustering

def sanitize(x):
    """
    Removes unwanted formatting characters from netlist strings.
    """
    return x.strip(',;\n ()')

def parser(file_):
    """
    Parses an ISCAS benchmark file (.bench) to extract circuit topology.
    Returns lists of inputs, outputs, internal gates, and wire connections.
    """
    with open(file_, 'r') as bench_file:
        input_nodes = []
        output_nodes = []
        gates = []
        wires = []

        for line in bench_file:
            line = line.strip()
            # Identify Input ports
            if line.startswith("INPUT"):
                node = sanitize(line.lstrip("INPUT"))
                input_nodes.append(node)
            # Identify Output ports
            elif line.startswith("OUTPUT"):
                node = sanitize(line.lstrip("OUTPUT"))
                output_nodes.append(node)
            # Parse logic gate definitions
            elif "=" in line:
                output_port, gate_definition = line.split("=")
                output_port = sanitize(output_port)
                gate_type, ports = gate_definition.strip().split("(")
                ports = [sanitize(x) for x in ports.split(",")]
                gate_name = output_port
                gates.append(gate_name)

                # Update wire source if it exists, otherwise create new
                existing_wire = next((w for w in wires if w[0] == output_port), None)
                if existing_wire:
                    existing_wire[1] = gate_name
                else:
                    wires.append([output_port, gate_name, []])

                # Update fan-out connections for input ports of the gate
                for i in ports:
                    existing_wire = next((w for w in wires if w[0] == i), None)
                    if existing_wire:
                        existing_wire[2].append(gate_name)
                    else:
                        # Wire format: [Node Name, Source, [Destinations]]
                        wires.append([i, i if i in input_nodes else ' ', [gate_name]])

    return input_nodes, output_nodes, gates, wires


def grapher(in_n, out_n, nodes, edges):
    """
    Constructs a NetworkX directed graph representation of the circuit.
    """
    G = nx.DiGraph()
    G.add_nodes_from(in_n)
    G.add_nodes_from(out_n)
    G.add_nodes_from(nodes)

    # Add edges based on parsed wire connections
    for i in edges:
        for j in i[2]:
            G.add_edge(i[1], j, weight=6)
    return G


def cluster_inputs_by_rare_nodes(graph, input_nodes, Rare_nodes, num_clusters=3):
    """
    Performs Spectral Clustering to group input nodes based on their
    connectivity relationship with identified rare nodes.
    """
    # Filter rare nodes to ensure they exist in the current graph
    valid_rare_nodes = [n for n in Rare_nodes if n in graph.nodes]

    # Create a subgraph containing only inputs and rare nodes
    sub_nodes = set(input_nodes).union(valid_rare_nodes)
    subgraph = graph.subgraph(sub_nodes)

    # Generate adjacency matrix and symmetrize it for spectral analysis
    adj_matrix = nx.to_numpy_array(subgraph)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2

    # Apply Spectral Clustering
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed').fit(adj_matrix)
    labels = clustering.labels_

    # Group input nodes by their assigned cluster label
    clustered_inputs = {}
    subgraph_nodes = list(subgraph.nodes)

    for idx, label in enumerate(labels):
        node = subgraph_nodes[idx]
        if node in input_nodes:
            if label not in clustered_inputs:
                clustered_inputs[label] = []
            clustered_inputs[label].append(node)

    return clustered_inputs