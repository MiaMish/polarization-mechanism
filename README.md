# Overview
This repository is accompanying a project that explores mechanisms for polarization reduction.  

# Pre-requisites
- Python 3.9

# Installation
- Clone the repository
- Install the requirements using `pip install -r requirements.txt`

# Usage
## Simulation
You can either run the simulations defined in the yaml files in the `src/resources` folder or use the cli tool.   
To run the simulations defined in the yaml files, run the following command:
```bash
python src/main.py
```

This will run all the simulations defined in the yaml files and save the results in the `resources` folder.  
You can tune the parameters of the simulations by editing the yaml files.  

The cli tool allows you to run a single simulation. To run the cli tool, run the following command:

```bash
# Crete a directory to store the results
mkdir -p "/path/to/results/directory/"

# Clear the old results (if exists)
./cli.py --base_db_path "/path/to/results/directory/" clear-db

# Add simulation configurations and store them
./cli.py --base_db_path "/path/to/results/directory/" append-configs --simulation_types SIMILARITY --mios 0.2,0.3 --mio_sigmas 0.075 --epsilons 0.2 --radical_exposure_etas None --switch_agent_rates None --switch_agent_sigmas None

# Run the simulations listed in the configs and store the results
./cli.py --base_db_path "/path/to/results/directory/" run-experiments

# Generate a combined results csv file
./cli.py --base_db_path "/path/to/results/directory/" generate-combined
```
You can see all the options available in the cli tool by running `./cli.py --help`.

## Analysis
The analysis is done using a script that runs on the combined results csv file. To run the analysis, run the following command:
```bash
python src/measurments_charts.py
```
You can modify the input file path and the output directory path in the script.  
The analysis will generate few charts based on the measurements in the combined results csv file.
