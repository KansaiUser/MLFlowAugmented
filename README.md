# Augmented MLFLow

## Data

Get the data from Kaggle website [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download) and save it inside the `data` folder

## Prepare

Install the poetry environment from the project base folder

```Shell
poetry install --no-root
```

## Run

Change folders and run the script

```Shell
cd src
poetry run python practice1.py
```

This script does not have any MLFlow function used

You can also run 

```Shell
 poetry run python practice1.py data.normal_frac=0.1
```

or for multiple runs

```Shell
poetry run python practice1.py -m data.normal_frac=0.1,0.2
```