# Topo Descriptor Experiments

This repository contains the code for experiments presented in:

> **B. T. Fasy, M. Makarchuk, S. Mika, and D. L. Millman**  
> *"Too Many or Too Few? Sampling Bounds for Topological Descriptors."*

These experiments study how different sampling strategies impact the use of topological descriptors, focusing on the challenges of oversampling, undersampling, and proposing an alternative method based on coarse stratification.

---

## Project Overview

The central theme of this project is **sampling** for topological descriptors.  
We explore two key issues:
- **Oversampling:** when geometric discretization produces too many descriptors.
- **Undersampling:** when fixed-size discretizations miss important strata.

Finally, we demonstrate the effectiveness of an **alternative approach** using coarse stratification on small plane graphs.

---

## Setup Instructions

1. Clone the repository and navigate into the project directory:

    ```bash
    git clone <repo_url>
    cd topo-descriptor-experiments
    ```

2. Create and activate the conda environment:

    ```bash
    conda env create -n your_conda_environment_name -f environment.yml
    conda activate your_conda_environment_name
    ```

3. Preprocess the datasets (RANDPTS, MPEG7, EMNIST):

    ```bash
    python preprocessing.py
    ```

---

## Running Main Experiments

### 1. Experiment 1: Too Many — Geometric-Based Discretization

**Goal:**  
Investigate how the size of the smallest one-stratum changes as the number of vertices increases.

**Run:**

    ```bash
    python exp_handler.py --epsilon 001 --experiment 1 --data 4
    python exp_handler.py --epsilon 005 --experiment 1 --data 4
    ```

- `epsilon` controls the ε-net resolution (`001` for 0.01 or `005` for 0.05).
- `data 4` runs across all datasets (RANDPTS, MPEG7, EMNIST).
- You can also run individually:
  - `--data 1` for RANDPTS
  - `--data 2` for MPEG7
  - `--data 3` for EMNIST

---

### 2. Experiment 2: Too Few — Constant Size Discretization

**Goal:**  
Study how fixing the discretization size affects the number of captured strata as the number of vertices varies.

**Run:**

    ```bash
    python exp_handler.py --epsilon 001 --experiment 2 --data 4
    python exp_handler.py --epsilon 005 --experiment 2 --data 4
    ```

- Same dataset options as in Experiment 1.

---

### 3. Experiment 3: Alternative Approach — Small Graphs

**Goal:**  
Demonstrate the use of coarse stratification for distinguishing small plane graphs.

**Run:**

    ```bash
    python small_graph_exp.py --bbox 60 --number_of_vertices 4
    python small_graph_exp.py --bbox 80 --number_of_vertices 5
    python small_graph_exp.py --bbox 30 --number_of_vertices 6
    ```

- `bbox` controls the bounding box size.
- `number_of_vertices` sets how many vertices are in the graph.

---

## Notes

- Results will be saved in different locations depending on the script — no unified `/results` folder is created.
- The `environment.yml` installs all necessary dependencies for graph processing and topological data analysis.
- Compatible with **Python 3.9.13** only!

