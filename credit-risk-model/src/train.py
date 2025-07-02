
# Task 5 - Model Training and Tracking (Enhanced Version)
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)
import mlflow
import mlflow.sklearn
import logging
from typing import Dict, Tuple
df_main = pd.read_csv('Final_feature_engineered_data.csv')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MLFLOW_EXPERIMENT_NAME = "Credit_Risk_Modeling"

class ModelTrainer:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()
        self.best_model = None
        self.best_metrics = {}
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def _split_data(self) -> Tuple:
        """Stratified train-test split to maintain class distribution"""
        return train_test_split(
            self.X, self.y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=self.y
        )

    @staticmethod
    def _get_hyperparameter_grids() -> Dict:
        """Define comprehensive hyperparameter grids for each model"""
        return {
            "logistic_regression": {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced']
            },
            "random_forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            },
            "gradient_boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'min_samples_split': [2, 5]
            }
        }

    def _evaluate_model(self, y_true, y_pred, y_proba) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "classification_report": classification_report(y_true, y_pred)
        }

    def train_model(self, model_name: str) -> Tuple[float, str]:
        """Train and evaluate a single model with hyperparameter tuning"""
        try:
            # Model selection
            models = {
                "logistic_regression": LogisticRegression(random_state=RANDOM_STATE),
                "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
                "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
            }
            
            if model_name not in models:
                raise ValueError(f"Unsupported model: {model_name}")

            # Setup MLflow run
            with mlflow.start_run(run_name=model_name, nested=True):
                # Hyperparameter tuning
                grid_search = GridSearchCV(
                    estimator=models[model_name],
                    param_grid=self._get_hyperparameter_grids()[model_name],
                    cv=StratifiedKFold(n_splits=CV_FOLDS),
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                logger.info(f"Starting GridSearchCV for {model_name}...")
                grid_search.fit(self.X_train, self.y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(self.X_test)
                y_proba = best_model.predict_proba(self.X_test)[:, 1]
                
                # Evaluate
                metrics = self._evaluate_model(self.y_test, y_pred, y_proba)
                
                # Log artifacts
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics({k: v for k, v in metrics.items() if k != "classification_report"})
                mlflow.sklearn.log_model(best_model, "model")
                
                logger.info(f"\n{model_name.upper()} Results:")
                logger.info(f"Best Params: {grid_search.best_params_}")
                logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
                logger.info(f"Classification Report:\n{metrics['classification_report']}")
                
                return metrics['roc_auc'], mlflow.active_run().info.run_id
                
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise

    def train_all_models(self):
        """Train and compare multiple models"""
        best_score = -1
        best_run_id = ""
        
        for model_name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            try:
                current_score, run_id = self.train_model(model_name)
                if current_score > best_score:
                    best_score = current_score
                    best_run_id = run_id
                    self.best_model = model_name
                    self.best_metrics = self._evaluate_model(
                        self.y_test,
                        mlflow.sklearn.load_model(f"runs:/{run_id}/model").predict(self.X_test),
                        mlflow.sklearn.load_model(f"runs:/{run_id}/model").predict_proba(self.X_test)[:, 1]
                    )
            except Exception as e:
                logger.warning(f"Skipping {model_name} due to error: {str(e)}")
                continue
        
        # Register best model
        if best_run_id:
            self._register_best_model(best_run_id)

    def _register_best_model(self, run_id: str):
        """Register the best model in MLflow Model Registry"""
        try:
            model_uri = f"runs:/{run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name="CreditRiskModel"
            )
            logger.info(f"\n✅ Best model registered:")
            logger.info(f"Name: {registered_model.name}")
            logger.info(f"Version: {registered_model.version}")
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Metrics: {self.best_metrics}")
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise

# Example Usage
if __name__ == "__main__":
    # Assuming df_main is loaded with features and 'is_high_risk' target
    X = df_main.drop(columns=['is_high_risk'])
    y = df_main['is_high_risk']
    
    trainer = ModelTrainer(X, y)
    trainer.train_all_models()