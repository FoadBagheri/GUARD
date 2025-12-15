# GUARD: An ABC-GA Hybrid Approach Utilizing Machine Learning and Dimensionality Reduction for Hardware Trojan Detection

This repository contains the official implementation of the paper:
**"GUARD: An ABC-GA Hybrid Approach Utilizing Machine Learning and Dimensionality Reduction for Hardware Trojan Detection"**.

## Abstract
The design and fabrication of integrated circuits (ICs) increasingly rely on outsourcing, involving multiple third-party entities. This multi-tiered supply chain introduces vulnerabilities, enabling adversarial actors to implant Hardware Trojans (HTs) at various stages of development. Such malicious modifications can lead to a wide range of security threats, including information leakage (e.g., MOLES Trojan) and denial-of-service (rarely triggered bit flip).

Although various methods exist for generating tests to detect HTs, they face significant challenges due to the immense complexity of the search space. This complexity renders the process impractical and results in insufficient trigger coverage. Effective HT detection depends on using suitable test vectors.

This paper introduces a novel algorithm, termed **GUARD**, which synergistically combines the **Artificial Bee Colony (ABC)** algorithm and **Genetic Algorithm (GA)**, along with a machine learning-based clustering technique for dimensionality reduction, within an automated framework for HT detection. Experimental evaluations on the ISCAS-85 and ISCAS-89 benchmark suites demonstrate a substantial increase in stimulus coverage and a marked reduction in execution time compared to existing state-of-the-art techniques.

## Repository Structure
The repository is organized as follows:

* **Main Algorithms:**
    * `ABC_GA_clustering_comb.py`: Main script for testing Combinational circuits.
    * `ABC_GA_clustering_seq.py`: Main script for testing Sequential circuits.
* **Helper Modules:**
    * `clustering_combinational.py`: Implements dimensionality reduction logic for combinational circuits.
    * `clustering_sequential.py`: Implements dimensionality reduction logic for sequential circuits.
    * `Dalgebra.py`: Helper module contains functions for boolean operations.
* **Datasets:**
    * `combinational_datasets/`: Contains ISCAS-85 benchmark files.
    * `sequential_datasets/`: Contains ISCAS-89 benchmark files.

## Installation
To run this code, you need to install the required dependencies.

1. Clone this repository:
   ```bash
   git clone [https://github.com/FoadBagheri/GUARD.git](https://github.com/FoadBagheri/GUARD.git)
   ```

2. Navigate to the directory:
   ```bash
   cd GUARD
   ```

3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
This repository provides separate implementations for Combinational and Sequential circuits. The scripts automatically utilize the datasets located in the respective folders.

### 1. For Combinational Circuits
To run the algorithm on combinational benchmarks (e.g., ISCAS-85), execute:
```bash
python ABC_GA_clustering_comb.py
```

### 2. For Sequential Circuits
To run the algorithm on sequential benchmarks (e.g., ISCAS-89), execute:
```bash
python ABC_GA_clustering_seq.py
```

## Key Features
* **Hybrid Metaheuristic:** Combines Artificial Bee Colony (ABC) and Genetic Algorithms (GA).
* **Dimensionality Reduction:** Uses ML-based clustering to handle complex search spaces.
* **Benchmarks:** Evaluated on ISCAS-85 and ISCAS-89 suites.

## Contact
If you have any questions regarding the code or the paper, please feel free to contact the authors.
