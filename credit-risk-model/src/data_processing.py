import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFMCalculator(BaseEstimator, TransformerMixin):
    """Calculate RFM features from transaction data"""
    
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date or datetime.now()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(X['TransactionStartTime']):
                X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
                
            # Group by customer
            rfm = X.groupby('CustomerId').agg({
                'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,
                'TransactionId': 'count',
                'Value': ['sum', 'mean']
            })
            
            # Flatten multi-index columns
            rfm.columns = ['Recency', 'Frequency', 'MonetaryTotal', 'MonetaryMean']
            
            # Add additional monetary features
            rfm['MonetaryMax'] = X.groupby('CustomerId')['Value'].max()
            rfm['MonetaryMin'] = X.groupby('CustomerId')['Value'].min()
            rfm['MonetaryStd'] = X.groupby('CustomerId')['Value'].std()
            
            return rfm.reset_index()
            
        except Exception as e:
            logger.error(f"Error calculating RFM: {str(e)}")
            raise

class HighRiskLabeler(BaseEstimator, TransformerMixin):
    """Create high-risk labels using KMeans clustering on RFM features"""
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        
    def fit(self, X, y=None):
        try:
            # Select and scale RFM features
            rfm_features = X[['Recency', 'Frequency', 'MonetaryTotal']]
            scaled_features = self.scaler.fit_transform(rfm_features)
            
            # Cluster customers
            self.kmeans.fit(scaled_features)
            return self
            
        except Exception as e:
            logger.error(f"Error fitting HighRiskLabeler: {str(e)}")
            raise
    
    def transform(self, X):
        try:
            # Select and scale RFM features
            rfm_features = X[['Recency', 'Frequency', 'MonetaryTotal']]
            scaled_features = self.scaler.transform(rfm_features)
            
            # Predict clusters
            clusters = self.kmeans.predict(scaled_features)
            
            # Identify high-risk cluster (cluster with highest recency, lowest frequency/monetary)
            cluster_stats = pd.DataFrame({
                'Cluster': clusters,
                'Recency': rfm_features['Recency'],
                'Frequency': rfm_features['Frequency'],
                'MonetaryTotal': rfm_features['MonetaryTotal']
            }).groupby('Cluster').mean()
            
            high_risk_cluster = cluster_stats.sort_values(
                by=['Recency', 'Frequency', 'MonetaryTotal'],
                ascending=[False, True, True]
            ).index[0]
            
            # Create binary label
            X['is_high_risk'] = (clusters == high_risk_cluster).astype(int)
            return X
            
        except Exception as e:
            logger.error(f"Error transforming HighRiskLabeler: {str(e)}")
            raise

def create_feature_pipeline():
    """Create full feature engineering pipeline"""
    
    # Numerical features pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Column transformers
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, ['Recency', 'Frequency', 'MonetaryTotal', 
                              'MonetaryMean', 'MonetaryMax', 'MonetaryMin', 'MonetaryStd']),
        ('cat', cat_pipeline, ['ProductCategory', 'ChannelId'])
    ])
    
    # Full pipeline
    pipeline = Pipeline([
        ('rfm', RFMCalculator()),
        ('labeler', HighRiskLabeler()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline