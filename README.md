# Non supervised Learning 

In this project for our master's course "unsupervised learning", we aim to reproduce some of the results of the article *"Learning from missing data with the binary latent block model."*

## Installation

### Creation of a conda environment
In order to implement the algorithms, we created some guided notebooks. Notebooks require a kernel with already installed packages. We propose to create a conda environment for this particular project using the following command:

```
conda create --name NSA_FVRL --file env_requirements.txt
```

### Kernel selection in python notebook
Before runing the notebook, we will need 

## Usage

To run the model on the dataset, use the script *run_on_dataset_parliament.py*:
```bash
python run_on_dataset_parliament.py
```
The default number of row classes is 3 and column classes is 5.



To run with a GPU use the argument *device* and specify the cuda index of desired gpu (often 0):
```bash
python run_on_dataset_parliament.py --device=0
```

To run with higher number of classes, use the arguments *nb_row_classes* and *nb_col_classes* as:
```bash
python run_on_dataset_parliament.py --nb_row_classes=3 --nb_col_classes=5
```

With higher number of classes, the memory of your GPU may overflow. In that case, you can use a second GPU with the argument *device2* (index cuda needs to be specify):

```bash
python run_on_dataset_parliament.py --device=0 --device2=1 --nb_row_classes=3 --nb_col_classes=8
```


The script can be keyboard interrupted  at any moment. In that case, the algorithm returns the MPs and texts classes and a plot of the voting matrix re-ordereded according to class memberships.

## License
[MIT]
