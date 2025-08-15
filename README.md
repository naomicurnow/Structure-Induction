# Kemp and Tenenbaum (2008) Structure Induction

This is an implementation of the model of structural induction presented in Kemp and Tenenbaum (2008) The discovery of structural form (doi 10.1073 pnas.0802631105). 

The aim of this model is to discover the underlying graph structure which best explains a given dataset. It works at two levels of abstraction: alongside finding the best *type* of graph (tree, ring, ...), it also finds the best *version* of that type (e.g., where are the branching points of the tree?). Formally, given a dataset ...

## Set up before running

It is recommended that you create a new Conda environment to run this project. To do this, use

```bash
conda create -n ENV_NAME
conda activate ENV_NAME
```

Then install dependencies using

```bash
pip install -r requirements.txt
```

## Tunable parameters

There are some ways to alter how the model runs:

### a) The forms it tests

This has functionality to test the following graph forms: `partition`, `order`, `chain`, `ring`, `hierarchy`, `tree`, `grid`, or `cylinder`. Currently, it is only possible to test undirected structures for each graph. You can modify which graph types are tested in `run model.py` by editing which forms are in the `forms` list:

```python
forms = [
    PartitionForm(),
    ChainForm(),
    TreeForm(),
    # OrderForm(), RingForm(), HierarchyForm(),
    # GridForm(), CylinderForm(),
]
```

### b) The number of times it restarts

As the model is non deterministic, it can get stuck in local optima or not search the space of structures sufficiently to find a well scoring structure of a particular type. Thus, it can be desirable to restart the search multiple times. The code here is automatically set to restart 10 times (as in the original Kemp and Tenenbaum paper), however you can alter this number in `config.py`:

```python
N_RESTARTS = # here
```

## Expected data format

This project currently accepts only `.mat` files. There are many of these provided as examples in the `/data` folder in the repo. Data can be one of two types:

### a) A feature matrix
This is where you have entities (e.g., animals) and a list of features about each (e.g., do they have a tail, are they stripy ...). Thus, the data should be shape `number of entities x number of features`. An example of this type of data is:
 
```matlab
% 3 entities × 2 features
X = [1.2 3.4;
     0.9 2.1;
     4.7 0.3];
names = {'A','B','C'};
save('my_features.mat','X','names');
```

Relevant files in `/data` of this type are the synthetic data `synth*.mat` and `animals.mat`. To run these examples, use 

```bash
python run_model.py --file FILENAME.mat
```

### b) A similarity matrix
This is where you have entities (e.g., cities) and a measure of how similar they are to all other entities in the set (e.g., how close they are geographically). The data should be a symmetric matrix of shape `number of entities x number of entities`. An example of this type of data is:
 
```matlab
% 3 entities × 3 entities (must be symmetric, diagonal = self-similarity)
S = [1.0 0.7 0.2;
     0.7 1.0 0.1;
     0.2 0.1 1.0];
names = {'A','B','C'};
save('my_similarity.mat','S','names');
```

Relevant files in `/data` of this type `cities.mat`, `colors.mat` and `faces.mat`. To run these examples, make sure to include the `--use_similarity` tag:

```bash
python run_model.py --use_similarity --file FILENAME.mat
```

## Other CLI tags
Other tags which may be of note:

- `--plot_entity_labels` can be used to modify the plots which output after the model is run (see below). By default, the plots render with black dots representing entities, however in some cases it can be desirable to print the entity labels instead if they have meaning (for example for the `animals.mat` dataset). This tag replaces the black dots with the labels.

## Expected outputs

Once the model has finished, there are two main sources to assess its findings:

### a) Logging

Every run writes two log files into `logs/`. It is recommended you look first at `logs/run-YYYYMMDD-HHMMSS-important.log`. Here, towards the end, you can find `Scores (raw)` which are the best log likelihoods found for each graph type, `Relative scores` which have been shifted so the graph type with the highest log likelihood has score 0 and the rest are negative relative to this, and `Softmax probabilities` which are softmaxed probabilites that each graph type underlies the data. Below this, there is a list of each of the graph types in order of their score (best scoring to least scoring), with the best scoring structure of that graph type printed.

For more information, `logs/run-YYYYMMDD-HHMMSS.log` contains the full history of the model fitting, with each structure that it searched through.

### b) Plotting

The code also outputs some plots automatically, saved in `plots/{FILENAME}/`. Here, you can find three images `all_graphs_v*.png` which contain the best scoring structure for each graph type alongside a bar chart of the final scores. The bar-plot axis is shifted so that the lowest score is approximately zero. The difference between the plots is that they use different methods to set the layout of each graph structure: `v1` fixes all entity positions to those used by the best-scoring graph and draws edges of the other graphs under this constraint; `v2` allows each structure to form freely; and `v3` also allows free layouts but attempts to minimise edge overlap for readability. Which layout “looks best” depends on the dataset and the particular run, so all three are provided for reference.

Additionally, the single best scoring graph is displayed in a figure window using Plotly. This allows interaction with the graph and ability to zoom, which can aid in understanding the graph structure when it is busy.

NOTE: I am still playing around with the parameters of the plotting functions, and so (especially when using entity labels, and especially for large datasets) the plots are not always great. I recommend running `demo_ring_feat.py` for some pretty plots.

## Start here!

It is recommended that you start by running one of the `demo` files: `demo_chain_feat.mat`, `demo_ring_feat.mat` or `demo_tree_feat.mat`. These are small files so they are relatively fast, and should finish in the order of seconds (depending on the number of forms you test and number of restarts). If these work well, you could then try one of the larger datasets, such as `animals.mat` or `colors.mat` (don't forget the `--use_similarity` tag for the latter!!). Expect these to take the order of minutes to hours. 
