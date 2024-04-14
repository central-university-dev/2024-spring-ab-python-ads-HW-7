import os
import pandas as pd
from catboost import CatBoostClassifier
from sklift.models import SoloModel, TwoModels
from sklift.datasets import fetch_x5
from joblib import dump
from sklearn.model_selection import train_test_split
from typing import Any


def fetch_and_prepare_data(storage_feature_path: str, storage_train_path: str) -> None:
    """
    Retrieves and processes the dataset then stores it into specified parquet files.

    Args:
        storage_feature_path (str): Destination path for the features dataframe.
        storage_train_path (str): Destination path for the training data dataframe.
    """
    dataset_bundle = fetch_x5()
    client_info = dataset_bundle.data["clients"].set_index("client_id")
    training_data = pd.concat(
        [dataset_bundle.data["train"], dataset_bundle.treatment, dataset_bundle.target], axis=1
    ).set_index("client_id")

    client_features = client_info.copy()
    client_features['initial_issue_time'] = (
        pd.to_datetime(client_features['first_issue_date']) - pd.Timestamp('1970-01-01')
    ) // pd.Timedelta('1s')
    client_features['initial_redeem_time'] = (
        pd.to_datetime(client_features['first_redeem_date']) - pd.Timestamp('1970-01-01')
    ) // pd.Timedelta('1s')
    client_features['delay_between_issue_redeem'] = (
        client_features['initial_redeem_time'] - client_features['initial_issue_time']
    )
    client_features.drop(['first_issue_date', 'first_redeem_date'], axis=1, inplace=True)

    client_features.to_parquet(storage_feature_path)
    training_data.to_parquet(storage_train_path)


def configure_model(config_type: str = 'solo') -> Any:
    """
    Configures and returns a model according to the specified type.

    Args:
        config_type (str): The configuration type of the model, either 'solo' or 'two'.

    Returns:
        A model instance configured according to the given type.
    """
    cat_params = {'iterations': 20, 'random_state': 42}
    if config_type == 'solo':
        return SoloModel(estimator=CatBoostClassifier(**cat_params))
    else:
        return TwoModels(
            estimator_trmnt=CatBoostClassifier(**cat_params),
            estimator_ctrl=CatBoostClassifier(**cat_params),
            method='vanilla'
        )

def execute_model_training(configured_model: Any, feature_storage_path: str, train_storage_path: str, **kwargs) -> Any:
    """
    Executes the training of the given model using the data stored at the specified paths.

    Args:
        configured_model: The model to be trained.
        feature_storage_path (str): Path to the stored features parquet file.
        train_storage_path (str): Path to the stored training data parquet file.

    Returns:
        The trained model.
    """
    features = pd.read_parquet(feature_storage_path)
    training_set = pd.read_parquet(train_storage_path)
    learning_indices, _ = train_test_split(training_set.index, test_size=0.2)
    X_train = features.loc[learning_indices]
    y_train = training_set.loc[learning_indices, 'target']
    treatment_train = training_set.loc[learning_indices, 'treatment_flg']
    configured_model.fit(X_train, y_train, treatment_train, **kwargs)
    return configured_model

if __name__ == "__main__":
    """
    Orchestrate the entire process of data preparation, model configuration, training, and storage.
    """
    feature_file_path = 'data/df_features.parquet'
    training_file_path = 'data/df_train.parquet'
    if not os.path.exists(feature_file_path):
        fetch_and_prepare_data(feature_file_path, training_file_path)
    for model_type in ['solo', 'two']:
        model = configure_model(model_type)
        train_params = {
            'estimator_fit_params': {'cat_features': ['gender']} if model_type == 'solo' else {
                'estimator_trmnt_fit_params': {'cat_features': ['gender']},
                'estimator_ctrl_fit_params': {'cat_features': ['gender']}
            }
        }
        trained_model = execute_model_training(model, feature_file_path, training_file_path, **train_params)
        dump(trained_model, f"{model_type}_cb.joblib")
