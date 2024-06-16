## Instructions

1. Clone repo to local machine and cd into topo-descriptor-experiments

2. To load environment run:
    conda env create -n "your_conda_environment_name" -f environment.yml

3. In terminal to start preprocess run:
    python3 preprocessing.py

4. Once the graphs have been created for all data sets, the experiments may be run from the exp_handler.py
    Smallest stratum experiment: 
        python3 exp_handler.py --epsilon 001 --experiment 1 --data 4 (001 aproximation)
        or
        python3 exp_handler.py --epsilon 005 --experiment 1 --data 4 (005 aproximation)
    Uniform random sample experiment:
        python3 exp_handler.py --epsilon 001 --experiment 2 --data 4 (001 aproximation)
        or
        python3 exp_handler.py --epsilon 005 --experiment 2 --data 4 (005 aproximation)
    Small graphs experiment:
        python3 too_few.py --bbox 60 --number_of_vertices 4,
        python3 too_few.py --bbox 80 --number_of_vertices 5
        or
        python3 too_few.py --bbox 30 --number_of_vertices 6



