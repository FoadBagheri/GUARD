# GUARD: An ABC-GA Hybrid Approach Utilizing Machine Learning and Dimensionality Reduction for Hardware Trojan Detection

This repository contains the official implementation of the paper:
**"GUARD: An ABC-GA Hybrid Approach Utilizing Machine Learning and Dimensionality Reduction for Hardware Trojan Detection"**.

## Abstract
The design and fabrication of integrated circuits (ICs) increasingly rely on outsourcing, involving multiple third-party entities. This multi-tiered supply chain introduces vulnerabilities, enabling adversarial actors to implant Hardware Trojans (HTs) at various stages of development. Such malicious modifications can lead to a wide range of security threats, including information leakage (e.g., MOLES Trojan) and denial-of-service (rarely triggered bit flip).Although various methods exist for generating tests to detect HTs, they face significant challenges due to the immense complexity of the search space. This complexity renders the process impractical and results in insufficient trigger coverage. Effective HT detection depends on using suitable test vectors. This paper introduces a novel algorithm, termed **GUARD**, which synergistically combines the **Artificial Bee Colony (ABC)** algorithm and **Genetic Algorithm (GA)**, along with a machine learning-based clustering technique for dimensionality reduction, within an automated framework for HT detection. Experimental evaluations on the ISCAS-85 and ISCAS-89 benchmark suites demonstrate a substantial increase in stimulus coverage and a marked reduction in execution time compared to existing state-of-the-art techniques.

## Implementation Details
This project utilizes a hybrid metaheuristic approach (ABC-GA) to optimize test generation and dimensionality reduction for securing ICs against Hardware Trojans.

**Key Features:**
* **Hybrid Metaheuristic:** Combines Artificial Bee Colony and Genetic Algorithms.
* **Dimensionality Reduction:** Uses ML-based clustering to handle complex search spaces.
* **Benchmarks:** Evaluated on ISCAS-85 and ISCAS-89 suites.

## Code Release
⚠️ **Note:**
The full source code and datasets are currently being prepared and cleaned for public release. They will be made available here immediately upon the acceptance of the paper.

Stay tuned!
