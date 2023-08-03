## Instructions

1. Clone repo to local machine and cd into topo-descriptor-experiments
2. To load environment run:
    conda env create -n "your_conda_environment_name" -f environment.yml

3. In terminal to start jupyterlab run:
    jupyter-lab
    Note: You can exit out of jupyterlab with CNTRL + C in terminal.

4. In juypter-lab, open `preprocess`Experiment_Supplement.ipynb notebook
    * Please run all preprocessing cells in order to create graphs for experiment files
5. Once the graphs have been created for all data sets, the experiments may be run from the Jupyter notebook
6. All experiments may also be run from exp_handler.py, with command-line arguments doccumented in the notebook.

## Testing

Some of the modules, for example `topology` have tests.  To run the tests:

    python -m unittest topology.test


