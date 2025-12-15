############################################
# Authors: Mostafa Hosseini & Foad Bagheri #
#               12/08/2025                 #
############################################

import numpy as np
from random import randint
import random
from Dalgebra import *
from datetime import datetime
from clustering_sequential import parser, grapher, cluster_find_inputs

start = datetime.now()


class Parser:
    """
    Main class for parsing sequential circuits (.bench files), simulating logic,
    handling flip-flops, calculating switching activity, and running ABC-GA optimization.
    """

    def __init__(self, seed):
        # Initialize random seeds for reproducibility
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Data structures for circuit representation
        self.varIndex = {}
        self.varMap = {}
        self.nodeLevel = {}
        self.sortedNode = []
        self.inputList = []
        self.outputList = []

        # Simulation and optimization parameters
        self.size = 0
        self.NoItr = 1000  # Number of iterations for Monte Carlo simulation
        self.threshold1 = 0.1
        self.threshold2 = 0.2
        self.coverage_ratio = 0.8  # Target coverage percentage for rare gates
        self.flipFlops = {}  # Storage for DFF state management
        self.timeSteps = 10  # Number of cycles for sequential simulation

    def readFile(self, fileName):
        """
        Reads the benchmark file, cleans inputs, and triggers parsing.
        """
        with open(fileName, "r") as f:
            lines = f.readlines()
            # Remove comments and empty lines
            lines = [i for i in lines if i != '\n' and i[0] != '#']
            lines = [i.replace("\n", "").replace(" ", "") for i in lines]
            self.__loadCode(lines)
            return

    def __loadCode(self, lines):
        """
        Parses the netlist line by line to build the graph structure,
        identifying inputs, outputs, gates, and flip-flops.
        """
        counter = 1
        for code in lines:
            if code.strip() == '':
                continue
            # Handle INPUT and OUTPUT declarations
            elif code.casefold().find("=") == -1:
                var = code[code.index("(") + 1: code.index(")")]
                type = "INPUT" if (code.casefold().find("input") != -1) else "OUTPUT"
                if var in self.varMap:
                    # Handle case where an OUTPUT is also an INPUT (creates a buffer)
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
                    # Initialize new node
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
                # Handle GATE and FLIP-FLOP definitions
                var = code[:code.index("=")]
                gate = code[code.index("=") + 1: code.index("(")]
                inputs = code[code.index("(") + 1: code.index(")")].split(",")

                if gate == "DFF":
                    # Register flip-flop with initial state 0
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

        # Post-parsing processing
        self.__cirLevelization()
        self.__sortByLevel()
        self.__inputRand()  # Initial random simulation
        self.switchingActivity()  # Calculate initial SWA
        self.rare_net()  # Identify rare nodes
        return

    # --- Circuit Levelization Methods ---

    def __nodeIsLevelable(self, node):
        # Checks if node can be assigned a level; DFF logic handled separately
        if self.varMap[node][4] == "DFF":
            return self.varMap[node][5][0] in self.nodeLevel
        return all(inp in self.nodeLevel for inp in self.varMap[node][5])

    def __maxOfInp(self, inpLst):
        # Finds the maximum level among input nodes
        return max(self.nodeLevel[inp] for inp in inpLst)

    def __nodeLv(self, node):
        return self.nodeLevel[node]

    def __sortByLevel(self):
        # Sorts nodes topologically to ensure correct simulation order
        self.sortedNode = sorted(self.nodeLevel.keys(), key=self.__nodeLv)
        self.inputList = sorted(self.inputList, key=self.__nodeLv)
        return

    def __cirLevelization(self):
        """
        Assigns logical levels to each node for topological sorting.
        """
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
        return

    def printSystem(self):
        """
        Displays the circuit specification table.
        """
        print("\nSystem Specification: ")
        print("|------|-------|-------|-------|---------|--------------|--------------|")

        label = "|{:^6}|{:^7}|{:^7}|{:^7}|{:^9}|{:^14}|{:^14}|".format(
            "NODE", "LEVEL", "TYPE", "VALUE", "GATE", "INPUT", "SWA"
        )
        print(label)
        print("|------|-------|-------|-------|---------|--------------|--------------|")

        for key in self.sortedNode:
            line = self.varMap[key]
            node = line[1]
            level = "inf" if node not in self.nodeLevel else self.nodeLevel[node]
            type = line[2]
            value = line[3]
            gate = line[4]
            SWA = str(line[10])
            input = ','.join([str(key) for key in line[5]]) if line[5] else "_____"

            printLine = "|{:{f}^6}|{:{f}^7}|{:{f}^7}|{:{f}^7}|{:{f}^9}|{:{f}^14}|{:{f}^14}|".format(
                node, level, type, value, gate, input, SWA, f='_')
            print(printLine)
        return

    def __clearValue(self):
        # Reset node values and flip-flop states before new simulation
        for var in self.varMap:
            self.varMap[var][3] = "_"
        for ff in self.flipFlops:
            self.flipFlops[ff]["state"] = "0"
        return

    def circuitSimulation(self, inputVector):
        """
        Runs sequential logic simulation over defined time steps.
        """
        self.__clearValue()
        for _ in range(self.timeSteps):
            self.__initInput(inputVector)
            self.__updateFlipFlops()
            for var in self.sortedNode:
                if var in self.inputList:
                    continue
                gate = self.varMap[var][4]
                inputs = self.varMap[var][5]
                result = self.__operate(var, gate, inputs)
                self.varMap[var][3] = result
        return

    def __initInput(self, inputVector):
        # Assign values to input nodes
        for i in range(len(self.inputList)):
            node = self.inputList[i]
            value = inputVector[i]
            self.varMap[node][3] = value
        return

    def __updateFlipFlops(self):
        """
        Updates the state of all flip-flops based on their inputs.
        """
        for ff, info in self.flipFlops.items():
            input_val = self.varMap[info["input"]][3]
            if input_val in ["0", "1"]:
                self.varMap[ff][3] = info["state"]
                info["state"] = input_val
        return

    def __operate(self, nodeOut, gate, inputNodes):
        # Performs the logic operation for a specific gate
        inputs = []
        for node in inputNodes:
            canonical = nodeOut + "_" + node
            inputName = canonical if canonical in self.varMap else node
            inputs.append(self.varMap[inputName][3])
        if gate == "NOT":
            return NOT(inputs)
        elif gate == "AND":
            return AND(inputs)
        elif gate == "OR":
            return OR(inputs)
        elif gate == "NAND":
            return NAND(inputs)
        elif gate == "NOR":
            return NOR(inputs)
        elif gate == "XOR":
            return XOR(inputs)
        elif gate == "XNOR":
            return XNOR(inputs)
        elif gate == "BUFF":
            return BUFF(inputs)
        elif gate == "DFF":
            return inputs[0]

    def __inputRand(self):
        # Generate random inputs for initial testing
        for _ in range(self.NoItr):
            inputVector = [str(randint(0, 1)) for _ in self.inputList]
            self.circuitSimulation(inputVector)
        return

    def switchingActivity(self):
        """
        Monte Carlo simulation to estimate Switching Activity (SWA) for all nodes.
        """
        SwResult = {}
        prevState = {}
        changeCount = {}
        # Initialize counters
        for node in self.sortedNode:
            SwResult[node] = [0, 0]
            changeCount[node] = {"0_to_1": 0, "1_to_0": 0}
            prevState[node] = None

        # Run random vector simulations
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
        """
        Identifies nodes (including DFFs) with switching activity in the target range.
        """
        gate_out_lst = ["AND", "NAND", "OR", "NOR", "XOR", "XNOR", "BUFF", "NOT", "DFF"]
        rare_list = []
        for node in self.sortedNode:
            if self.varMap[node][4] in gate_out_lst:
                # Calculate average activity
                rare_node = (self.varMap[node][10][0] + self.varMap[node][10][1]) / self.NoItr
                if self.threshold1 < rare_node < self.threshold2:
                    rare_list.append([node, rare_node])
        return rare_list

    def ABC_GA(self, cluster0, cluster1, cluster2):
        """
        Hybrid Artificial Bee Colony (ABC) and Genetic Algorithm (GA) optimization
        to generate test vectors that activate rare gates.
        """
        num_bees = 100
        mutation_rate = 0.1
        crossover_rate = 0.8
        max_trials = 5
        rare_gates = self.rare_net()
        total_rare_gates = len(rare_gates)
        activated_rare_gates = set()
        test_vectors = []
        total_test_vectors_generated = 0

        def get_activated_rare_gates(position):
            # Helper to check which rare gates are triggered by a vector
            self.circuitSimulation(position)
            activated = set()
            for rare_gate in rare_gates:
                gate_name = rare_gate[0]
                if int(self.getValue(gate_name)) == 1:
                    activated.add(gate_name)
            return activated

        # Initialize bee positions based on clustered inputs
        bee_positions = []
        combined_clusters = cluster0 + cluster1 + cluster2
        sorted_effective_inputs = sorted([(idx, 0) for idx in combined_clusters], key=lambda x: x[1], reverse=True)

        for _ in range(num_bees):
            position = np.array(['0'] * len(self.inputList))
            for idx, _ in sorted_effective_inputs:
                position[idx] = str(randint(0, 1))
            for idx in range(len(position)):
                if idx not in dict(sorted_effective_inputs).keys():
                    position[idx] = str(randint(0, 1))
            fitness = self.fitness_function(position, cluster0, cluster1, cluster2)
            bee_positions.append({'position': position, 'fitness': fitness, 'trials': 0})

        global_best_position = max(bee_positions, key=lambda x: x['fitness'])['position']
        global_best_fitness = max(bee_positions, key=lambda x: x['fitness'])['fitness']

        # Main optimization loop
        while len(activated_rare_gates) < self.coverage_ratio * total_rare_gates:
            total_test_vectors_generated += 1
            print(f"\nTest vector generation attempt: {total_test_vectors_generated}")

            # Phase 1: Employed Bees (Local Search)
            for i, bee in enumerate(bee_positions):
                neighbor_idx = randint(0, num_bees - 1)
                while neighbor_idx == i:
                    neighbor_idx = randint(0, num_bees - 1)
                neighbor = bee_positions[neighbor_idx]
                new_position = self.crossover(bee['position'], neighbor['position'])[0]
                new_position = self.mutate(new_position, mutation_rate, dict(sorted_effective_inputs).keys())
                new_fitness = self.fitness_function(new_position, cluster0, cluster1, cluster2)
                if new_fitness > bee['fitness']:
                    bee['position'] = new_position
                    bee['fitness'] = new_fitness
                    bee['trials'] = 0
                else:
                    bee['trials'] += 1

            # Phase 2: Onlooker Bees (Genetic Operations)
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

                    # Store vectors that activate new rare gates
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

            # Phase 3: Scout Bees (Abandon stuck solutions)
            for bee in bee_positions:
                if bee['trials'] > max_trials:
                    new_position = np.array(['0'] * len(self.inputList))
                    for idx, _ in sorted_effective_inputs:
                        new_position[idx] = str(randint(0, 1))
                    for idx in range(len(new_position)):
                        if idx not in dict(sorted_effective_inputs).keys():
                            new_position[idx] = str(randint(0, 1))
                    bee['position'] = new_position
                    bee['fitness'] = self.fitness_function(new_position, cluster0, cluster1, cluster2)
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

    def fitness_function(self, position, cluster0, cluster1, cluster2):
        """
        Evaluates the solution quality based on induced switching activity in rare nodes.
        """
        inputVector_full = ['0'] * len(self.inputList)
        for idx in cluster0 + cluster1 + cluster2:
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

        # Calculate SWA for the given pair of vectors
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

    def SWA_Alpha(self):
        """
        Analyzes the final switching activity distribution against thresholds.
        """
        SW_list = []
        for node in self.sortedNode:
            node_Alpha = self.varMap[node][10][0] + self.varMap[node][10][1]
            SW_list.append(node_Alpha)
            Alpha_list = [x / self.NoItr for x in SW_list]
        alph = sum(1 for i in Alpha_list if self.threshold1 < i <= self.threshold2)
        print("Alpha: ", alph)
        print("Threshold1: ", self.threshold1)
        print("Threshold2: ", self.threshold2)
        return Alpha_list

    def getValue(self, node):
        return self.varMap[node][3]


# Main execution block
if __name__ == "__main__":
    ob = Parser(seed=42)
    ob.readFile('sequential_datasets/s5378.bench')

    # Parse file for graph structure and perform clustering
    in_n, out_n, nodes, edges = parser('sequential_datasets/s5378.bench')
    G = grapher(in_n, out_n, nodes, edges)

    rare_net_data = ob.rare_net()

    rare_nodes = [item[0] for item in rare_net_data]

    # Cluster inputs based on their influence on rare nodes
    clustered_inputs = cluster_find_inputs(G, in_n, rare_nodes, num_clusters=3)

    Cluster0 = clustered_inputs.get(0, [])
    Cluster1 = clustered_inputs.get(1, [])
    Cluster2 = clustered_inputs.get(2, [])

    ob.printSystem()
    test_vectors = ob.ABC_GA(Cluster0, Cluster1, Cluster2)
    print("Execution time: ", (datetime.now() - start).total_seconds())
    ob.SWA_Alpha()