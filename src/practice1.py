import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import RocCurveDisplay, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from loguru import logger
import os
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load data
    df = pd.read_csv(cfg.data.path)
    df = df.drop("Time", axis=1)
    print(df.head())
    print(f"Shape: {df.shape}")
    normal = df[df.Class == 0].sample(frac=cfg.data.normal_frac, random_state=cfg.model.random_state).reset_index(drop=True)
    anomaly = df[df.Class == 1]

    print(f"Normal: {normal.shape}")
    print(f"Anomalies: {anomaly.shape}")

    # Data Splitting
    normal_train, normal_test = train_test_split(normal, test_size=cfg.data.test_size, random_state=cfg.model.random_state)
    anomaly_train, anomaly_test = train_test_split(anomaly, test_size=cfg.data.test_size, random_state=cfg.model.random_state)
    normal_train, normal_validate = train_test_split(normal_train, test_size=cfg.data.validate_size, random_state=cfg.model.random_state)
    anomaly_train, anomaly_validate = train_test_split(anomaly_train, test_size=cfg.data.validate_size, random_state=cfg.model.random_state)

    # Create combined sets
    x_train = pd.concat((normal_train, anomaly_train))
    x_test = pd.concat((normal_test, anomaly_test))
    x_validate = pd.concat((normal_validate, anomaly_validate))
    y_train = np.array(x_train["Class"])
    y_test = np.array(x_test["Class"])
    y_validate = np.array(x_validate["Class"])
    x_train = x_train.drop("Class", axis=1)
    x_test = x_test.drop("Class", axis=1)
    x_validate = x_validate.drop("Class", axis=1)

    # Data Scaling
    scaler = StandardScaler()
    scaler.fit(pd.concat((normal, anomaly)).drop("Class", axis=1))
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_validate = scaler.transform(x_validate)

    # Model Training and Evaluation
    sk_model = LogisticRegression(
        random_state=cfg.model.random_state,
        max_iter=cfg.model.max_iter,
        solver=cfg.model.solver
    )
    train(sk_model, x_train, y_train)
    evaluate(sk_model, x_test, y_test, cfg.paths)

def train(sk_model, x_train, y_train):
    sk_model = sk_model.fit(x_train, y_train)
    train_acc = sk_model.score(x_train, y_train)
    logger.info(f"Train Accuracy: {train_acc:.3%}")

def evaluate(sk_model, x_test, y_test, paths):
    eval_acc = sk_model.score(x_test, y_test)
    preds = sk_model.predict(x_test)
    auc_score = roc_auc_score(y_test, preds)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    
    print(f"AUC Score: {auc_score:.3%}")
    print(f"Eval Accuracy: {eval_acc:.3%}")
    print(f"Precision: {precision:.3%}")
    print(f"Recall: {recall:.3%}")

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Output dir {output_dir}")

    # print(f"Working directory : {os.getcwd()}")
    # print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    
    # ROC Curve
    RocCurveDisplay.from_estimator(sk_model, x_test, y_test, name='Scikit-learn ROC Curve')
    roc_path = os.path.join(output_dir, paths.roc_plot)
    plt.savefig(roc_path)
    # plt.show()
    # plt.clf()
    plt.close()
    print("ROC curve saved")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, preds)
    ax = sns.heatmap(conf_matrix, annot=True, fmt='g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix")
    conf_matrix_path = os.path.join(output_dir,paths.conf_matrix)
    plt.savefig(conf_matrix_path)
    plt.close() 
    # plt.savefig(paths.conf_matrix)
    print("Confusion matrix saved")

if __name__ == "__main__":
    main()
