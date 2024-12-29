import os
import sys
import warnings
import optuna
from catboost import CatBoostClassifier
import mlflow
import pandas as pd
from model.train import train_model
from preprocessing.prepare_data import prepare_data, get_pool
from utils.utils import plot_feature_importance, validate_model, plot_shap
from optuna.visualization import plot_parallel_coordinate, plot_optimization_history
warnings.filterwarnings('ignore')

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# data config
data_path = "data/data.csv"

# params
RANDOM_STATE = 42
TEST_SIZE = 0.2

# model path to save
model_path = "artefacts/catboost_model.pkl"

EXPERIMENT_NAME = 'optuna_catboost_experiment'

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)


def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'verbose': False
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    accuracy = validate_model(model, X_test, y_test)
    return accuracy

if __name__ == '__main__':
    
    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = prepare_data(data, TEST_SIZE, RANDOM_STATE)
    print(TEST_SIZE)
    
    # Convert the data to CatBoost Pool format
    train_data = get_pool(X_train, y_train)
    test_data = get_pool(X_test, y_test)
    
    with mlflow.start_run():
         # log data
        dataset = mlflow.data.from_pandas(X_train)
        mlflow.log_input(dataset, context='fraud data')
        
        # Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=30)
        
        # Log best parameters
        best_params = study.best_params
        mlflow.log_params(best_params)
        
        # Train best model
        best_model = train_model(X_train, y_train, best_params, train_data)
        
        # Validate and log metrics
        accuracy = validate_model(best_model, X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # plot
        fig = plot_feature_importance(best_model)
        print(fig)
        mlflow.log_figure(fig, "feature_importance.png")
        
        curve = plot_shap(best_model, X_train)
        print(curve)
        mlflow.log_figure(curve, "shap.png")
        
        parallel_plot = plot_parallel_coordinate(study)
        parallel_plot.write_html("parallel_coordinates.html")
        mlflow.log_artifact("parallel_coordinates.html", "optuna_plots")

        optimization_history_plot = plot_optimization_history(study)
        optimization_history_plot.write_html("optimization_history.html")
        mlflow.log_artifact("optimization_history.html", "optuna_plots")
        
        # Save the best model
        mlflow.catboost.log_model(best_model, "model")

        print(f"Best parameters: {best_params}")
        print(f"Test Accuracy: {accuracy}")

