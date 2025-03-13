# Physics-informed-training-of-normalizing-flows

## Clone the repository

```shell script
git clone --recursive https://github.com/StefanWahl/Physics-informed-training-of-normalizing-flows.git
cd Physics-informed-training-of-normalizing-flows
```

## Linkin folders for data and results

Use this to link your local location for the experimental results and data sets.

```shell script
ln -s <Your_local_data_folder> data
ln -s <Your_local_results_folder> results
```

## Create a new environment

```shell script
conda create -n pinf python=3.10.
conda activate pinf
```

## Install the package and required packages:

```shell script
pip install -r requirements.txt
pip install -e .
```

## Install additional packages

The following repositories have to be installed:

* https://github.com/vislearn/FFF

Install this package in development mode:

* https://github.com/vislearn/FrEIA


## Generate training data

For the low-dimensional data sets use the files provided in the folder `./create_data/`. For the scalar lattice theory, proceed as follows. First run a simulation to generate new data. This can for example done as follows:

```shell script
cd Applying-Energy-Based-Models-on-the-Ising-model-and-a-scalar-lattice-field-theory-in-two-dimensions/ScalarTheory/
python3 Simulation_Scalar_Theory.py --N 8 --kappa_min 0.2 --kappa_max 0.4 --dkappa 0.01 --n_iter 10000000 --record 1 --seed 0
```

After the the simulation has terminated, return to the root folder of the repository and postprocess the recorded data.

```shell script
python3 create_data/ScalarTheory_create_combined_datasets.py --source ./Applying-Energy-Based-Models-on-the-Ising-model-and-a-scalar-lattice-field-theory-in-two-dimensions/ScalarTheory/ScalarTheory/N_8_LANGEVIN_SPECIFIC_Data_Set/ --destination ./data/ScalarTheory/training_data/
```

Afterwards, remove the original results of the simulations. In total, for a given lattice size, three runs have to be conducted: One to generate a training set (Use `./data/ScalarTheory/training_data/` as destination), One to generate a validation set (Use `./data/ScalarTheory/validation_data/` as destination) and one to generate a reference simulation. For the reference simulation, set `--record 0` in the simulation and choose a smaller `--dkappa`. Copy the result of this simulation directly into the folder `./data/ScalarTheory/training_data/`. Use different random seeds `--seed` for the generation of training and validation data.

## Trainin of normalizing flows

To train a new normalizing flow can be trained using `train.py`:


```shell script
python3 train.py --tag <Your experiment name> --config_path ./config/<data set name>/config_<experiment>.json
```

## Evaluation of traind models

To evaluate the trained models use the file `plot.ipynb` in the respective folder in `./evaluation/`. In the case of the evaluation of the models trained for the scalar lattice theory run the evaluation script in advance:

```shell script
python3 evaluation/ScalarTheory/evaluation_script_ScalarTheory.py --experiment_folder ./results/runs_ScalarTheory/<Your experiment name>/lightning_logs/version_0
```
