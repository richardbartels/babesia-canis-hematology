# Babesia
This repository contains the analysis code for the machine learning of
*Identification of parameters and formulation of a statistical and machine learning model to identify acute Babesia canis infections in dogs using available ADVIA hematology analyzer data* by Pijnacker et al.


## How to run this code
 
### Virtual environment
The easiest way to recreate the virtual environment is to use [Anaconda](https://www.anaconda.com).
Create and activate the virtual environment using:

```
conda env create -f environment.yml
conda activate babesia
```

### Train and evaluate models
***Note: data is not included with the repository.***

For training using hyperopt and mlflow use:
```python src/main.py --mode train```.

Results can be inspected by running `mlflow ui`.
After selecting the best hyperparameters update
`src/best_fit_parameters.py` and run
```python src/main.py --mode test```
to evaluate performance on the test set.

### DVC
[`dvc`](https://dvc.org/doc/start) is used for data and model version control to ensure reproducibility.