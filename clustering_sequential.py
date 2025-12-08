############################################
# Authors: Mostafa Hosseini & Foad Bagheri #
#               12/08/2025                 #
############################################

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
from random import randint
import random

def sanitize(x):
    return x.strip(',;\n ()')

file_ = "sequential_datasets/s5378.bench"

class CircuitSimulator:
    def __init__(self, seed=42, iterations=1000, threshold=0.1):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.varMap = {}
        self.varIndex = {}
        self.inputList = []
        self.outputList = []
        self.sortedNode = []
        self.nodeLevel = {}
        self.size = 0
        self.NoItr = iterations
        self.threshold = threshold
        self.flipFlops = {}

    def readFile(self, fileName):
        with open(fileName, "r") as f:
            lines = f.readlines()
            lines = [i for i in lines if i != '\n' and i[0] != '#']
            lines = [i.replace("\n", "").replace(" ", "") for i in lines]
            self.__loadCode(lines)

    def __loadCode(self, lines):
        counter = 1
        for code in lines:
            if code.strip() == '':
                continue
            elif code.casefold().find("=") == -1:
                var = code[code.index("(") + 1: code.index(")")]
                type = "INPUT" if (code.casefold().find("input") != -1) else "OUTPUT"
                if var in self.varMap:
                    if type == "OUTPUT" and self.varMap[var][2] == "INPUT":
                        newVar = var + 'o'
                        self.varMap[newVar] = [
                            counter, newVar, type, "_", "BUFF", [var], [], [1, 1], 0, [0, 0], "_"
                        ]
                        self.outputList.append(var)
                        counter += 1
                        self.size += 1
                        continue
                    self.varMap[var][2] = type
                    if type == "OUTPUT" and var not in self.outputList:
                        self.outputList.append(var)
                else:
                    self.varMap[var] = [counter, var, type, "_", "___", [], [], [-1, -1], 0, [0, 0], "_"]
                    self.varIndex[var] = counter
                    if type == "INPUT":
                        self.inputList.append(var)
                        self.varMap[var][7] = [1, 1]
                    else:
                        self.outputList.append(var)
                    counter += 1
                    self.size += 1
            else:
                var = code[:code.index("=")]
                gate = code[code.index("=") + 1: code.index("(")]
                inputs = code[code.index("(") + 1: code.index(")")].split(",")
                if gate == "DFF":
                    self.flipFlops[var] = {"input": inputs[0], "state": "0"}
                    self.varMap[var] = [
                        counter, var, "FLIPFLOP", "0", "DFF", inputs, [], [-1, -1], 0, [0, 0], "_"
                    ]
                    self.varIndex[var] = counter
                    counter += 1
                    self.size += 1
                else:
                    if var in self.varMap:
                        self.varMap[var][4] = gate
                        self.varMap[var][5] = inputs
                    else:
                        self.varMap[var] = [
                            counter, var, "___", "_", gate, inputs, [], [-1, -1], 0, [0, 0], "_"
                        ]
                        self.varIndex[var] = counter
                        counter += 1
                        self.size += 1
        self.__cirLevelization()
        self.__sortByLevel()

    def __nodeIsLevelable(self, node):
        if self.varMap[node][4] == "DFF":
            return self.varMap[node][5][0] in self.nodeLevel
        return all(inp in self.nodeLevel for inp in self.varMap[node][5])

    def __maxOfInp(self, inpLst):
        return max(self.nodeLevel[inp] for inp in inpLst)

    def __nodeLv(self, node):
        return self.nodeLevel[node]

    def __sortByLevel(self):
        self.sortedNode = sorted(self.nodeLevel.keys(), key=self.__nodeLv)
        self.inputList = sorted(self.inputList, key=self.__nodeLv)

    def __cirLevelization(self):
        isUpdate = True
        allAssigned = False
        while isUpdate and not allAssigned:
            isUpdate = False
            for line in self.varMap.values():
                node = line[1]
                if node in self.nodeLevel:
                    continue
                elif line[2] == "INPUT":
                    self.nodeLevel[node] = 0
                    isUpdate = True
                elif self.__nodeIsLevelable(node):
                    max_level = self.__maxOfInp(line[5])
                    self.nodeLevel[node] = 1 + max_level
                    isUpdate = True
                    if len(self.nodeLevel) == self.size:
                        allAssigned = True

    def __clearValue(self):
        for var in self.varMap:
            self.varMap[var][3] = "_"
        for ff in self.flipFlops:
            self.flipFlops[ff]["state"] = "0"

    def circuitSimulation(self, inputVector):
        self.__clearValue()
        for _ in range(10):  # Simulate over 10 time steps for sequential circuits
            self.__initInput(inputVector)
            self.__updateFlipFlops()
            for var in self.sortedNode:
                if var in self.inputList:
                    continue
                gate = self.varMap[var][4]
                inputs = self.varMap[var][5]
                result = self.__operate(var, gate, inputs)
                self.varMap[var][3] = result

    def __initInput(self, inputVector):
        for i in range(len(self.inputList)):
            node = self.inputList[i]
            value = inputVector[i]
            self.varMap[node][3] = value

    def __updateFlipFlops(self):
        for ff, info in self.flipFlops.items():
            input_val = self.varMap[info["input"]][3]
            if input_val in ["0", "1"]:
                self.varMap[ff][3] = info["state"]
                info["state"] = input_val

    def __operate(self, nodeOut, gate, inputNodes):
        inputs = []
        for node in inputNodes:
            canonical = nodeOut + "_" + node
            inputName = canonical if canonical in self.varMap else node
            inputs.append(self.varMap[inputName][3])
        if gate == "NOT":
            return '1' if inputs[0] == '0' else '0'
        elif gate == "AND":
            return '1' if all(i == '1' for i in inputs) else '0'
        elif gate == "OR":
            return '1' if any(i == '1' for i in inputs) else '0'
        elif gate == "NAND":
            return '0' if all(i == '1' for i in inputs) else '1'
        elif gate == "NOR":
            return '0' if any(i == '1' for i in inputs) else '1'
        elif gate == "XOR":
            return '1' if sum(1 for i in inputs if i == '1') % 2 == 1 else '0'
        elif gate == "XNOR":
            return '1' if sum(1 for i in inputs if i == '1') % 2 == 0 else '0'
        elif gate == "BUFF":
            return inputs[0]
        elif gate == "DFF":
            return inputs[0]

    def switchingActivity(self):
        SwResult = {}
        prevState = {}
        changeCount = {}
        for node in self.sortedNode:
            SwResult[node] = [0, 0]
            changeCount[node] = {"0_to_1": 0, "1_to_0": 0}
            prevState[node] = None
        for _ in range(self.NoItr):
            inputVector = [str(randint(0, 1)) for _ in self.inputList]
            self.circuitSimulation(inputVector)
            for node in self.sortedNode:
                currentValue = int(self.getValue(node))
                if prevState[node] is not None:
                    if prevState[node] == 0 and currentValue == 1:
                        changeCount[node]["0_to_1"] += 1
                    elif prevState[node] == 1 and currentValue == 0:
                        changeCount[node]["1_to_0"] += 1
                prevState[node] = currentValue
                SWA0_1 = [changeCount[node]["0_to_1"], changeCount[node]["1_to_0"]]
                self.varMap[node][10] = SWA0_1
        return changeCount

    def rare_net(self):
        gate_out_lst = ["AND", "NAND", "OR", "NOR", "XOR", "XNOR", "BUFF", "NOT", "DFF"]
        rare_list = []
        for node in self.sortedNode:
            if self.varMap[node][4] in gate_out_lst:
                rare_node = (self.varMap[node][10][0] + self.varMap[node][10][1]) / self.NoItr
                if rare_node < self.threshold:
                    rare_list.append(node)
        return rare_list

    def getValue(self, node):
        return self.varMap[node][3]

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
                gate_type = gate_type.strip().lower()
                ports = [sanitize(x) for x in ports.split(",")]
                gate_name = output_port
                gates.append(gate_name)
                if gate_type == "dff":
                    input_port = ports[0]
                    existing_wire = next((w for w in wires if w[0] == output_port), None)
                    if existing_wire:
                        existing_wire[1] = gate_name
                    else:
                        wires.append([output_port, gate_name, []])
                    if input_port in input_nodes:
                        existing_wire = next((w for w in wires if w[0] == input_port), None)
                        if existing_wire:
                            existing_wire[2].append(gate_name)
                        else:
                            wires.append([input_port, input_port, [gate_name]])
                    else:
                        existing_wire = next((w for w in wires if w[0] == input_port), None)
                        if existing_wire:
                            existing_wire[2].append(gate_name)
                        else:
                            wires.append([input_port, ' ', [gate_name]])
                else:
                    existing_wire = next((w for w in wires if w[0] == output_port), None)
                    if existing_wire:
                        existing_wire[1] = gate_name
                    else:
                        wires.append([output_port, gate_name, []])
                    for i in ports:
                        if i in input_nodes:
                            existing_wire = next((w for w in wires if w[0] == i), None)
                            if existing_wire:
                                existing_wire[2].append(gate_name)
                            else:
                                wires.append([i, i, [gate_name]])
                        else:
                            existing_wire = next((w for w in wires if w[0] == i), None)
                            if existing_wire:
                                existing_wire[2].append(gate_name)
                            else:
                                wires.append([i, ' ', [gate_name]])
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

def cluster_find_inputs(graph, input_nodes, rare_nodes, num_clusters=3):
    # Subgraph containing rare nodes and their influencers
    subgraph_nodes = set(rare_nodes)
    for node in rare_nodes:
        predecessors = nx.ancestors(graph, node) | {node}
        subgraph_nodes.update(predecessors & set(input_nodes))
    subgraph = graph.subgraph(subgraph_nodes)
    if len(subgraph.nodes) == 0:
        return {i: [] for i in range(num_clusters)}
    adj_matrix = nx.to_numpy_array(subgraph)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed').fit(adj_matrix)
    labels = clustering.labels_
    clustered_inputs = {}
    node_list = list(subgraph.nodes)
    input_indices = {node: idx for idx, node in enumerate(graph.nodes)}
    for idx, label in enumerate(labels):
        node = node_list[idx]
        if node in input_nodes:
            if label not in clustered_inputs:
                clustered_inputs[label] = []
            clustered_inputs[label].append(input_indices[node])
    return clustered_inputs

# Run the code
sim = CircuitSimulator(seed=42)
sim.readFile(file_)
sim.switchingActivity()
rare_nodes = sim.rare_net()
in_n, out_n, nodes, edges = parser(file_)
G = grapher(in_n, out_n, nodes, edges)
clustered_inputs = cluster_find_inputs(G, in_n, rare_nodes, num_clusters=3)
for cluster, input_indices in clustered_inputs.items():
    print(f"Cluster{cluster} = {input_indices}")