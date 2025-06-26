import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.data_processing import RFMCalculator, HighRiskLabeler

@pytest.fixture
def sample_data():
    data = {
        'CustomerId': [1, 1, 2, 2, 2, 3],
        'TransactionStartTime': [
            datetime.now() - timedelta(days=10),
            datetime.now() - timedelta(days=5),
            datetime.now() - timedelta(days=20),
            datetime.now() - timedelta(days=15),
            datetime.now() - timedelta(days=2),
            datetime.now() - timedelta(days=30)
        ],
        'TransactionId': [101, 102, 201, 202, 203, 301],
        'Value': [100, 200, 50, 150, 300, 75]
    }
    return pd.DataFrame(data)

def test_rfm_calculator(sample_data):
    """Test RFM feature calculation"""
    snapshot_date = datetime.now()
    rfm_calc = RFMCalculator(snapshot_date)
    result = rfm_calc.transform(sample_data)
    
    assert not result.empty
    assert 'Recency' in result.columns
    assert 'Frequency' in result.columns
    assert 'MonetaryTotal' in result.columns
    assert len(result) == 3  # 3 unique customers

def test_high_risk_labeler(sample_data):
    """Test high risk labeling"""
    snapshot_date = datetime.now()
    rfm_calc = RFMCalculator(snapshot_date)
    rfm_data = rfm_calc.transform(sample_data)
    
    labeler = HighRiskLabeler(random_state=42)
    labeler.fit(rfm_data)
    result = labeler.transform(rfm_data)
    
    assert 'is_high_risk' in result.columns
    assert result['is_high_risk'].isin([0, 1]).all()
    assert result['is_high_risk'].sum() > 0  # At least one high risk