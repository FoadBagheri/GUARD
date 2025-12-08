############################################
# Authors: Mostafa Hosseini & Foad Bagheri #
#               12/08/2025                 #
############################################

import numpy as np
from random import randint
from scoap import *
import random
from Dalgebra import *
from datetime import datetime
from clustering import parser, grapher, cluster_inputs_by_rare_nodes

start = datetime.now()


class Parser:
    def __init__(self, seed):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.varIndex = {}
        self.varMap = {}
        self.nodeLevel = {}
        self.sortedNode = []
        self.inputList = []
        self.outputList = []
        self.size = 0
        self.NoItr = 1000
        self.threshold1 = 0.1
        self.threshold2 = 0.2

        self.Coverage_ratio = 0.8

    def readFile(self, fileName):
        with open(fileName, "r") as f:
            lines = f.readlines()
            lines = [i for i in lines if i != '\n' and i[0] != '#']
            lines = [i.replace("\n", "").replace(" ", "") for i in lines]
            self.__loadCode(lines)
            return

    def __loadCode(self, lines):
        counter = 1
        for code in lines:
            if code.strip() == '':
                continue
            # INPUT/OUTPUT
            elif code.casefold().find("=") == -1:
                var = code[code.index("(") + 1: code.index(")")]
                type = "INPUT" if (code.casefold().find("input") != -1) else "OUTPUT"
                if var in self.varMap:
                    # I/O  node type Buffer
                    if type == "OUTPUT" and self.varMap[var][2] == "INPUT":
                        newVar = var + 'o'
                        self.varMap[newVar] = [
                            counter, newVar, type, "_", "BUFF", [var], [], [1, 1], 0, [0, 0], "_"
                        ]
                        self.outputList.append(var)
                        counter += 1
                        self.size += 1
                        continue
                        # END I/O
                    self.varMap[var][2] = type
                    if type == "OUTPUT" and type not in self.outputList:
                        self.outputList.append(var)
                else:
                    self.varMap[var] = [counter, var, type, "_", "___", [], [], [-1, -1], 0, [0, 0], "_"
                                        ]
                    self.varIndex[var] = counter
                    if type == "INPUT":
                        self.inputList.append(var)
                        self.varMap[var][7] = [1, 1]
                    else:
                        self.outputList.append(var)
                    counter += 1
                    self.size += 1
            else:
                # gate
                var = code[:code.index("=")]
                gate = code[code.index("=") + 1: code.index("(")]
                inputs = code[code.index("(") + 1: code.index(")")].split(",")

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
        self.__genScoap()
        self.__inputRand()
        self.switchingActivity()
        self.HTS_1()
        self.rare_net()
        return

    # Circuit Levelization
    def __nodeIsLevelable(self, node):
        return all(inp in self.nodeLevel for inp in self.varMap[node][5])

    # max level of input
    def __maxOfInp(self, inpLst):
        return max(self.nodeLevel[inp] for inp in inpLst)

    # create sorted node by level
    def __nodeLv(self, node):
        return self.nodeLevel[node]

    def __sortByLevel(self):
        self.sortedNode = sorted(self.nodeLevel.keys(), key=self.__nodeLv)
        self.inputList = sorted(self.inputList, key=self.__nodeLv)
        return

    # circuit Levelization
    def __cirLevelization(self):
        isUpdate = True
        allAssigned = False
        while (isUpdate and not allAssigned):
            isUpdate = False
            for line in self.varMap.values():
                node = line[1]
                if node in self.nodeLevel:
                    continue
                elif line[2] == "INPUT":
                    self.nodeLevel[node] = 0
                    isUpdate = True
                elif self.__nodeIsLevelable(node):
                    max = self.__maxOfInp(line[5])
                    self.nodeLevel[node] = 1 + max
                    isUpdate = True
                    if len(self.nodeLevel) == self.size:
                        allAssigned = True
        return

    def __genScoap(self):
        for var in self.sortedNode:
            if self.varMap[var][2] == "INPUT":
                continue
            gate = self.varMap[var][4]
            inputs = self.varMap[var][5]
            self.varMap[var][7] = self.__operateSCOAP(var, gate, inputs)
        return

    def __operateSCOAP(self, nodeOut, gate, inputNodes):
        inputs = []
        for node in inputNodes:
            canonical = nodeOut + "_" + node
            inputName = canonical if canonical in self.varMap else node
            inputs.append(self.varMap[inputName][7])

        if gate == "NOT":
            return scoapNOT(inputs)
        elif gate == "AND":
            return scoapAND(inputs)
        elif gate == "OR":
            return scoapOR(inputs)
        elif gate == "NAND":
            return scoapNAND(inputs)
        elif gate == "NOR":
            return scoapOR(inputs)
        elif gate == "XOR":
            return scoapXOR(inputs)
        elif gate == "XNOR":
            return scoapXNOR(inputs)
        elif gate == "BUFF":
            return scoapBUFF(inputs)

    def printSystem(self):
        print()
        print("SystemSpecification: ")
        print("|-----|------|-------|------|---------|------------|---------|---------|--------------|---------|")
        label = "|{:>5}|{:>6}|{:>7}|{:>6}|{:^9}|{:^12}|{:^9}|{:^9}|{:^14}|{:<14}".format(
            "node", "level", "type", "val", "gate", "(CC0, CC1)", "(CO)", "MC(0, 1)", "input", "SWA"
        )
        print(label)
        print("|-----|------|-------|------|---------|------------|---------|---------|--------------|---------|")

        for key in self.sortedNode:
            line = self.varMap[key]
            node = line[1]
            level = "inf" if node not in self.nodeLevel else self.nodeLevel[node]
            type = line[2]
            val = line[3]
            gate = line[4]
            cntr = ("(" + ",".join([str(key) for key in line[7]]) + ")") if line[7] != [-1 - 1] else "(u, u)"
            CO_ = str(line[8])
            SWA = str(line[10])
            MC = ("(" + ",".join([str(key) for key in line[9]]) + ")") if line[9] != [0, 0] else "[0, 0]"
            input = ','.join([str(key) for key in line[5]]) if line[5] else "_____"
            printLine = "|{:{f}>5}|{:{f}>6}|{:{f}>7}|{:{f}>6}|{:{f}>9}|{:{f}^12}|{:{f}^9}|{:{f}^9}|{:{f}^14}|{:{f}^14}|".format(
                node, level, type, val, gate, cntr, CO_, MC, input, SWA, f='_')
            print(printLine)
        return

    def printInOut(self):
        print("# of input list", len(self.inputList))
        print("# of output list", len(self.outputList))
        return

    def __clearValue(self):
        for var in self.varMap:
            self.varMap[var][3] = "_"
        return

    def circuitSimulation(self, inputVector):
        self.__clearValue()
        self.__initInput(inputVector)
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
        return

    def __operate(self, nodeOut, gate, inputNodes):
        inputs = []
        for node in inputNodes:
            canonical = nodeOut + "_" + node
            inputName = canonical if canonical in self.varMap else node
            inputs.append(self.varMap[inputName][3])
        if gate == "NOT":
            return NOT(inputs)
        if gate == "AND":
            return AND(inputs)
        if gate == "OR":
            return OR(inputs)
        if gate == "NAND":
            return NAND(inputs)
        if gate == "NOR":
            return NOR(inputs)
        if gate == "XOR":
            return XOR(inputs)
        if gate == "XNOR":
            return XNOR(inputs)
        if gate == "BUFF":
            return BUFF(inputs)

    def __inputRand(self):
        for _ in range(self.NoItr):
            inputVector = [str(randint(0, 1)) for _ in self.inputList]
        self.circuitSimulation(inputVector)
        return

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
                SWA0_1 = []
                currentValue = int(self.getValue(node))

                if prevState[node] is not None:
                    if prevState[node] == 0 and currentValue == 1:
                        changeCount[node]["0_to_1"] += 1
                    elif prevState[node] == 1 and currentValue == 0:
                        changeCount[node]["1_to_0"] += 1
                prevState[node] = currentValue
                SWA0_1.extend([changeCount[node]["0_to_1"], changeCount[node]["1_to_0"]])
                self.varMap[node][10] = SWA0_1
        return

    def rare_net(self):
        gate_out_lst = ["AND", "NAND", "OR", "NOR", "XOR", "XNOR", "BUFF", "NOT"]
        rare_list = []
        for node in self.sortedNode:
            if self.varMap[node][4] in gate_out_lst:
                rare_node = (self.varMap[node][10][0] + self.varMap[node][10][1]) / self.NoItr
                if rare_node < self.threshold1:
                    rare_list.append([node, rare_node])
        return rare_list

    def ABC_GA(self, cluster1, cluster2, cluster3):
        num_bees = 20
        mutation_rate = 0.1
        crossover_rate = 0.8
        max_trials = 5
        rare_gates = self.rare_net()
        total_rare_gates = len(rare_gates)
        activated_rare_gates = set()
        test_vectors = []
        total_test_vectors_generated = 0

        def get_activated_rare_gates(position):
            self.circuitSimulation(position)
            activated = set()
            for rare_gate in rare_gates:
                gate_name = rare_gate[0]
                if int(self.getValue(gate_name)) == 1:
                    activated.add(gate_name)
            return activated

        # Initialize bee positions
        bee_positions = []
        combined_clusters = [(idx, val) for idx, val in enumerate(cluster1 + cluster2 + cluster3)]
        sorted_effective_inputs = sorted(combined_clusters, key=lambda x: x[1], reverse=True)

        for _ in range(num_bees):
            position = np.array(['0'] * len(self.inputList))
            for idx, _ in sorted_effective_inputs:
                position[idx] = str(randint(0, 1))
            for idx in range(len(position)):
                if idx not in dict(sorted_effective_inputs).keys():
                    position[idx] = str(randint(0, 1))

            fitness = self.fitness_function(position, cluster1, cluster2, cluster3)
            bee_positions.append({'position': position, 'fitness': fitness, 'trials': 0})

        global_best_position = max(bee_positions, key=lambda x: x['fitness'])['position']
        global_best_fitness = max(bee_positions, key=lambda x: x['fitness'])['fitness']

        while len(activated_rare_gates) < self.Coverage_ratio * total_rare_gates:
            total_test_vectors_generated += 1
            print(f"\nTest vector generation attempt: {total_test_vectors_generated}")

            # Employed Bees Phase
            for i, bee in enumerate(bee_positions):
                neighbor_idx = randint(0, num_bees - 1)
                while neighbor_idx == i:
                    neighbor_idx = randint(0, num_bees - 1)
                neighbor = bee_positions[neighbor_idx]

                new_position = self.crossover(bee['position'], neighbor['position'])[0]
                new_position = self.mutate(new_position, mutation_rate, dict(sorted_effective_inputs).keys())
                new_fitness = self.fitness_function(new_position, cluster1, cluster2, cluster3)

                if new_fitness > bee['fitness']:
                    bee['position'] = new_position
                    bee['fitness'] = new_fitness
                    bee['trials'] = 0
                else:
                    bee['trials'] += 1

            # Onlooker Bees Phase
            for _ in range(num_bees):
                parent1 = max(bee_positions, key=lambda x: x['fitness'])['position']
                parent2 = bee_positions[randint(0, num_bees - 1)]['position']
                if np.random.rand() < crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1, mutation_rate, dict(sorted_effective_inputs).keys())
                    child2 = self.mutate(child2, mutation_rate, dict(sorted_effective_inputs).keys())

                    activated_by_child1 = get_activated_rare_gates(child1)
                    activated_by_child2 = get_activated_rare_gates(child2)

                    print(
                        f"Test vector {total_test_vectors_generated} (ABC-GA) activated rare gates: {activated_by_child1}")
                    new_rare_gates_child1 = activated_by_child1 - activated_rare_gates
                    if new_rare_gates_child1:
                        print(f"New rare gates activated by child1: {new_rare_gates_child1}")
                        activated_rare_gates.update(new_rare_gates_child1)
                        test_vectors.append(child1)

                    new_rare_gates_child2 = activated_by_child2 - activated_rare_gates
                    if new_rare_gates_child2:
                        print(f"New rare gates activated by child2: {new_rare_gates_child2}")
                        activated_rare_gates.update(new_rare_gates_child2)
                        test_vectors.append(child2)

                    print(f"Total rare gates activated so far: {len(activated_rare_gates)}/{total_rare_gates}")

            # Scout Bees Phase
            for bee in bee_positions:
                if bee['trials'] > max_trials:
                    new_position = np.array(['0'] * len(self.inputList))
                    for idx, _ in sorted_effective_inputs:
                        new_position[idx] = str(randint(0, 1))
                    for idx in range(len(new_position)):
                        if idx not in dict(sorted_effective_inputs).keys():
                            new_position[idx] = str(randint(0, 1))
                    bee['position'] = new_position
                    bee['fitness'] = self.fitness_function(new_position, cluster1, cluster2, cluster3)
                    bee['trials'] = 0

        print(f"\nGlobal best test vector: {global_best_position}")
        print(f"Global best fitness: {global_best_fitness}")
        print(f"\nMinimum number of tests needed to activate all rare gates: {len(test_vectors)}")
        print(f"Total number of test vectors generated: {total_test_vectors_generated}")
        return test_vectors

    def mutate(self, child, mutation_rate, effective_inputs):
        for i in effective_inputs:
            if np.random.rand() < mutation_rate:
                child[i] = '1' if child[i] == '0' else '0'
        return child

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def fitness_function(self, position, cluster1, cluster2, cluster3):
        inputVector_full = ['0'] * len(self.inputList)
        for idx in cluster1 + cluster2 + cluster3:
            inputVector_full[idx] = position[idx]

        SwResult = {}
        prevState = {}
        changeCount = {}
        rare_lst = self.rare_net()
        for node in self.sortedNode:
            SwResult[node] = [0, 0]
            changeCount[node] = {"0_to_1": 0, "1_to_0": 0}
            prevState[node] = None

        inputVector_init = ['0'] * len(self.inputList)
        pairList = [[inputVector_init, inputVector_full]]

        for pair in pairList:
            changeCount = {node: {"0_to_1": 0, "1_to_0": 0} for node in self.sortedNode}
            prevState = {node: None for node in self.sortedNode}
            for inp_vec in pair:
                self.circuitSimulation(inp_vec)
                swa_lst = []
                for node in rare_lst:
                    SWA0_1 = []
                    currentvalue = int(self.getValue(node[0]))
                    if prevState[node[0]] is not None:
                        if prevState[node[0]] == 0 and currentvalue == 1:
                            changeCount[node[0]]["0_to_1"] += 1
                        elif prevState[node[0]] == 1 and currentvalue == 0:
                            changeCount[node[0]]["1_to_0"] += 1
                    prevState[node[0]] = currentvalue
                    SWA0_1.append([changeCount[node[0]]["0_to_1"], changeCount[node[0]]["1_to_0"]])
                    swa_lst.extend(SWA0_1)
                node_swa_lst = [sum(pair) for pair in swa_lst]
                fitness_values = sum(node_swa_lst)
        return fitness_values

    def HTS_1(self):
        HTS_list = []
        for node in self.sortedNode:
            HTS = abs(ob.varMap[node][7][0] - ob.varMap[node][7][1]) / max(ob.varMap[node][7][0], ob.varMap[node][7][1])
            HTS_list.append(HTS)
        return HTS_list

    def SWA_Alpha(self):
        SW_list = []
        for node in self.sortedNode:
            node_Alpha = ob.varMap[node][10][0] + ob.varMap[node][10][1]
            SW_list.append(node_Alpha)
            Alpha_list = [x / self.NoItr for x in SW_list]
        alph = sum(1 for i in Alpha_list if i <= self.threshold1)
        print("alph: ", alph, "\n", "threshold: ", self.threshold1)
        return Alpha_list

    def perturb_vector(self, vector, index):
        perturbed_vector = vector.copy()
        perturbed_vector[index] = '1' if perturbed_vector[index] == '0' else '0'
        return perturbed_vector

    def perturb_vectors(self, vectors):
        all_perturbed_vectors = []
        for vector in vectors:
            perturbed_vectors = []
            for i in range(len(vector)):
                perturbed_vector = self.perturb_vector(vector, i)
                perturbed_vectors.append(perturbed_vector)
            all_perturbed_vectors.append(perturbed_vectors)
        return all_perturbed_vectors

    def getValue(self, node):
        return self.varMap[node][3]


# Main execution
if __name__ == "__main__":
    target_file = 'combinational_datasets/c880.bench'

    ob = Parser(seed=42)
    ob.readFile(target_file)
    ob.printSystem()

    # Get Rare Nets (names) from Parser
    rare_net_data = ob.rare_net()
    rare_node_names = [item[0] for item in rare_net_data]

    # Parse file for Clustering (requires file path)
    in_n, out_n, nodes, edges = parser(target_file)
    G = grapher(in_n, out_n, nodes, edges)

    # Perform Clustering using Rare Node Names
    clustered_inputs_raw = cluster_inputs_by_rare_nodes(G, in_n, rare_node_names, num_clusters=3)

    # Convert Cluster Node Names (Strings) to InputList Indices (Integers)
    # ABC_GA requires indices to access 'position' arrays.
    cluster_indices = {}
    for c_id, node_list in clustered_inputs_raw.items():
        indices = []
        for node_name in node_list:
            if node_name in ob.inputList:
                indices.append(ob.inputList.index(node_name))
        cluster_indices[c_id] = indices

    # Extract Clusters
    Cluster0 = cluster_indices.get(0, [])
    Cluster1 = cluster_indices.get(1, [])
    Cluster2 = cluster_indices.get(2, [])

    print(f"Generated Clusters (Indices):")
    print(f"C0: {len(Cluster0)}, C1: {len(Cluster1)}, C2: {len(Cluster2)}")

    ob.ABC_GA(Cluster0, Cluster1, Cluster2)
    print("time: ", (datetime.now() - start).total_seconds())
    ob.SWA_Alpha()