# Physics-informed-training-of-normalizing-flows

#Linking the data path

Use this to link your local location for the experimental results and data sets.

```shell script
cd Physics-informed-training-of-normalizing-flows
ln -s <Your_local_data_folder> data
ln -s <Your_local_results_folder> results
```

#Create a new environment

```shell script
conda create -n pinf python=3.10.
conda activate pinf
```

#Link folders

Use this to link your local location for the experimental results and data sets.

#Install the package:

```shell script
pip install -e .
```

#Install additional packages

The following repositories have to be installed:

* https://github.com/vislearn/FFF

Install this package in development mode:

* https://github.com/vislearn/FrEIA
