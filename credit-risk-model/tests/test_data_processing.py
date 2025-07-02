import pytest
from sklearn.datasets import make_classification
from model_trainer import ModelTrainer

@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)

def test_data_splitting(sample_data):
    X, y = sample_data
    trainer = ModelTrainer(X, y)
    assert len(trainer.X_train) == 800  # 80% of 1000
    assert len(trainer.X_test) == 200

def test_model_training(sample_data):
    X, y = sample_data
    trainer = ModelTrainer(X, y)
    roc_auc, _ = trainer.train_model("logistic_regression")
    assert 0 <= roc_auc <= 1  # ROC AUC should be between 0 and 1