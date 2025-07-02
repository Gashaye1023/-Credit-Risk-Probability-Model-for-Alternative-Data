# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from xverse.transformer import WOE # For WoE encoding
import json # For saving best params if needed

from src.data_processing import create_preprocessing_pipeline, create_proxy_target_variable, DateFeatureExtractor, AggregateFeatureCreator

class WOEEncoderTransformer(WOE, Pipeline):

    def __init__(self, cols=None, V_IV=True, **kwargs):
        super().__init__(V_IV=V_IV) # Initialize WOE
        self.cols = cols
        self.kwargs = kwargs

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("WOEEncoderTransformer requires a target variable (y) for fitting.")
        if self.cols is None:
            self.cols = X.columns.tolist()
        self.woe_transformer = WOE(V_IV=self.V_IV, **self.kwargs)
        self.woe_transformer.fit(X[self.cols], y)
        return self

    def transform(self, X):
        return self.woe_transformer.transform(X[self.cols])

def train_model():
    # --- MLflow Setup ---
    mlflow.set_experiment("Credit_Risk_Model_Training")

    with mlflow.start_run():
        # Log parameters (e.g., data paths, random states, model types)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_clusters_rfm", 3)
        mlflow.log_param("snapshot_date", '2023-12-31')


        # --- Load Data ---
        df_raw = pd.read_csv('Final_preprocessed_data.csv')


        # --- Task 4: Proxy Target Variable Engineering ---
        print("Creating proxy target variable...")
        customer_high_risk_labels = create_proxy_target_variable(df_raw, snapshot_date_str=mlflow.active_run().data.params['snapshot_date'],
                                                                 n_clusters=mlflow.active_run().data.params['n_clusters_rfm'],
                                                                 random_state=mlflow.active_run().data.params['random_state'])
        print(f"Number of high-risk customers: {customer_high_risk_labels['is_high_risk'].sum()}")

        # Merge high-risk labels back to the main DataFrame (after initial feature engineering)
        # First, apply initial feature engineering that doesn't depend on target
        date_extractor = DateFeatureExtractor(date_column='TransactionDate')
        df_temp_date = date_extractor.fit_transform(df_raw.copy())

        agg_creator = AggregateFeatureCreator(group_by_col='CustomerId', amount_col='TransactionAmount')
        df_processed_initial = agg_creator.fit_transform(df_temp_date.copy())

        # Merge the target variable
        df_merged = df_processed_initial.merge(customer_high_risk_labels, on='CustomerId', how='left')

        # Drop TransactionId as it's not a feature
        df_merged = df_merged.drop(columns=['TransactionId'])
        # Drop CustomerId for model training, but keep for merging if needed later
        df_model_ready = df_merged.drop(columns=['CustomerId'])


        # Separate features (X) and target (y)
        X = df_model_ready.drop(columns=['is_high_risk'])
        y = df_model_ready['is_high_risk']
        
        # Identify columns for preprocessing AFTER initial feature creation
        # Numerical columns include original and newly created ones
        numerical_features_final = X.select_dtypes(include=np.number).columns.tolist()
        # Categorical columns
        categorical_features_final = X.select_dtypes(include='object').columns.tolist()



        # --- Data Splitting ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=mlflow.active_run().data.params['test_size'],
                                                            random_state=mlflow.active_run().data.params['random_state'],
                                                            stratify=y) # Stratify for imbalanced target

        # --- Create a flexible preprocessing pipeline for ColumnTransformer ---
        # This will be part of the model pipeline for consistent transformation
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numerical_features_final),
                ('cat_ohe', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features_final),
            ],
            remainder='passthrough' # Keep other columns that might be added later if not transformed
        )

        # --- Model Selection and Training ---
        models = {
            'Logistic Regression': LogisticRegression(solver='liblinear', random_state=mlflow.active_run().data.params['random_state']),
            'Decision Tree': DecisionTreeClassifier(random_state=mlflow.active_run().data.params['random_state']),
            'Random Forest': RandomForestClassifier(random_state=mlflow.active_run().data.params['random_state']),
            'Gradient Boosting': GradientBoostingClassifier(random_state=mlflow.active_run().data.params['random_state'])
        }

        best_model_name = None
        best_model_performance = 0
        best_model_pipeline = None

        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            mlflow.log_param("model_name", name)

            # Create a full pipeline including preprocessing and the model
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            # --- Hyperparameter Tuning (Example for Logistic Regression) ---
            if name == 'Logistic Regression':
                param_grid = {
                    'classifier__C': [0.01, 0.1, 1, 10, 100],
                    'classifier__penalty': ['l1', 'l2']
                }
                search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
                search.fit(X_train, y_train)
                print(f"Best parameters for {name}: {search.best_params_}")
                mlflow.log_params({f"best_params_{name}": search.best_params_})
                current_model_pipeline = search.best_estimator_
            elif name == 'Gradient Boosting':
                param_dist = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7]
                }
                search = RandomizedSearchCV(model_pipeline, param_dist, n_iter=10, cv=3, scoring='roc_auc',
                                            random_state=mlflow.active_run().data.params['random_state'], n_jobs=-1)
                search.fit(X_train, y_train)
                print(f"Best parameters for {name}: {search.best_params_}")
                mlflow.log_params({f"best_params_{name}": search.best_params_})
                current_model_pipeline = search.best_estimator_
            else:
                current_model_pipeline = model_pipeline
                current_model_pipeline.fit(X_train, y_train)

            # --- Model Evaluation ---
            y_pred = current_model_pipeline.predict(X_test)
            y_proba = current_model_pipeline.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            }
            print(f"Metrics for {name}: {metrics}")
            mlflow.log_metrics(metrics)

            if roc_auc > best_model_performance:
                best_model_performance = roc_auc
                best_model_name = name
                best_model_pipeline = current_model_pipeline

        print(f"\nBest Model: {best_model_name} with ROC-AUC: {best_model_performance:.4f}")
        mlflow.set_tag("best_model", best_model_name)
        mlflow.log_metric("best_roc_auc", best_model_performance)


        # --- Register Best Model in MLflow Model Registry ---
        if best_model_pipeline:
            mlflow.sklearn.log_model(
                sk_model=best_model_pipeline,
                artifact_path="credit_risk_model",
                registered_model_name="CreditRiskModel",
                signature=mlflow.models.infer_signature(X_test, y_proba) # Infer signature from test data
            )
            print(f"Best model '{best_model_name}' registered to MLflow Model Registry.")


if __name__ == '__main__':
    train_model()