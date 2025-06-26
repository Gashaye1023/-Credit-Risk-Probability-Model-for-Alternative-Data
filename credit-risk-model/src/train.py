import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV
from .data_processing import create_feature_pipeline
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath):
    """Load and preprocess data"""
    try:
        df = pd.read_csv(filepath)
        pipeline = create_feature_pipeline()
        features = pipeline.fit_transform(df)
        
        # Get target variable
        target = pipeline.named_steps['labeler'].named_steps['high_risk_labeler'].is_high_risk
        
        # Get feature names
        num_features = ['Recency', 'Frequency', 'MonetaryTotal', 
                       'MonetaryMean', 'MonetaryMax', 'MonetaryMin', 'MonetaryStd']
        cat_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['ProductCategory', 'ChannelId'])
        feature_names = num_features + list(cat_features)
        
        return features, target, feature_names
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_models(X_train, y_train):
    """Train and evaluate multiple models"""
    
    # Define models and parameters for grid search
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }
    }
    
    best_model = None
    best_score = 0
    
    for model_name, config in models.items():
        with mlflow.start_run(run_name=model_name):
            # Log model parameters
            mlflow.log_params({'model': model_name})
            
            # Grid search
            grid = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            
            # Log best parameters and metrics
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("best_cv_score", grid.best_score_)
            
            # Track best model
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_
                
            # Log model
            mlflow.sklearn.log_model(grid.best_estimator_, model_name)
            
            logger.info(f"{model_name} - Best AUC: {grid.best_score_:.4f}")
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Log metrics to MLflow
        for name, value in metrics.items():
            mlflow.log_metric(f"test_{name}", value)
            
        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_metric("tn", cm[0][0])
        mlflow.log_metric("fp", cm[0][1])
        mlflow.log_metric("fn", cm[1][0])
        mlflow.log_metric("tp", cm[1][1])
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def main():
    # Initialize MLflow
    mlflow.set_experiment("Credit_Risk_Modeling")
    
    # Load and split data
    X, y, feature_names = load_data('../data/raw/xente_transactions.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    best_model = train_models(X_train, y_train)
    
    # Evaluate best model
    metrics = evaluate_model(best_model, X_test, y_test)
    logger.info(f"Best model metrics: {metrics}")
    
    # Register best model
    mlflow.sklearn.log_model(best_model, "best_model")
    mlflow.sklearn.save_model(best_model, "../models/best_model")
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()