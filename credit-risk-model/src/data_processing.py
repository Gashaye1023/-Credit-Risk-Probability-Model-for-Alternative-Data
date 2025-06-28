import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.datetime_col] = pd.to_datetime(X_copy[self.datetime_col], errors='coerce')
        X_copy['TransactionHour'] = X_copy[self.datetime_col].dt.hour
        X_copy['TransactionDay'] = X_copy[self.datetime_col].dt.day
        X_copy['TransactionMonth'] = X_copy[self.datetime_col].dt.month
        X_copy['TransactionYear'] = X_copy[self.datetime_col].dt.year
        return X_copy

class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, id_col='AccountId', aggregate_cols=['Amount', 'Value']):
        self.id_col = id_col
        self.aggregate_cols = aggregate_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        agg_features = X_copy.groupby(self.id_col)[self.aggregate_cols].agg(['sum', 'mean', 'count', 'std'])
        agg_features.columns = ['_'.join(col).strip() + '_per_customer' for col in agg_features.columns.values]
        X_copy = pd.merge(X_copy, agg_features, on=self.id_col, how='left')
        for col in self.aggregate_cols:
            std_col_name = f'{col}_std_per_customer'
            if std_col_name in X_copy.columns:
                X_copy[std_col_name] = X_copy[std_col_name].fillna(0)
        return X_copy

def get_feature_engineering_pipeline():
    initial_numerical_cols = ['Amount', 'Value']
    initial_categorical_cols = [
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
        'ProductCategory', 'ChannelId', 'PricingStrategy'
    ]
    datetime_generated_cols = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    agg_generated_cols = [
        'Amount_sum_per_customer', 'Amount_mean_per_customer',
        'Amount_count_per_customer', 'Amount_std_per_customer',
        'Value_sum_per_customer', 'Value_mean_per_customer',
        'Value_count_per_customer', 'Value_std_per_customer'
    ]
    all_numerical_features = initial_numerical_cols + datetime_generated_cols + agg_generated_cols
    all_categorical_features = initial_categorical_cols

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, all_numerical_features),
            ('cat', categorical_transformer, all_categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('datetime_features', DateTimeFeatureExtractor()),
        ('aggregate_features', CustomerAggregator()),
        ('preprocessing', preprocessor)
    ])

    return pipeline

if __name__ == '__main__':
    print("--- Feature Engineering Pipeline Example ---")
    df=pd.read_csv("./data/raw/data.csv")

    print("Original DataFrame head:")
    print(df.head())
    print("\nOriginal DataFrame info:")
    df.info()

    if 'FraudResult' in df.columns:
        X = df.drop('FraudResult', axis=1)
        y = df['FraudResult']
        print(f"\nTarget variable 'y' shape: {y.shape}")
    else:
        X = df.copy()
        y = None

    feature_pipeline = get_feature_engineering_pipeline()
    X_transformed = feature_pipeline.fit_transform(X)
    print(f"\nShape of transformed data: {X_transformed.shape}")
    print("\nTransformed data (first 5 rows, first 15 columns for brevity):")
    print(X_transformed[:5, :15])

    df.to_csv("./data/processed/processed_data.csv", index=False)
    X_transformed = pd.DataFrame(X_transformed, columns=feature_pipeline.named_steps['preprocessing'].get_feature_names_out())
    X_transformed.to_csv("./data/processed/processed_features.csv", index=False)
    