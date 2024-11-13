# Augmented MLFLow

## Data

Get the data from Kaggle website [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download) and save it inside the `data` folder

## Prepare

Install the poetry environment from the project base folder

```Shell
poetry install --no-root
```

## Run

Change folders and in one terminal run

```Shell
cd src
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

and in another terminal run the script

```Shell
cd src
poetry run python practiceMLFlow1.py
```

You can also run 

```Shell
 poetry run python practice1.py data.normal_frac=0.1
```

or for multiple runs

```Shell
poetry run python practice1.py -m data.normal_frac=0.1,0.2
```


Here by going to http://localhost:5000/ you will see the MLOps UI and can see the runs on the experiment `scikit_learn_experiment`
