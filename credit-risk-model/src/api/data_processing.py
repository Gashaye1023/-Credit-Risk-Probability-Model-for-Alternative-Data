import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
import warnings

# Suppress specific warnings for cleaner output during demonstration
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Custom Transformers for Feature Engineering ---

class AggregateFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, customer_id_col='CustomerId', transaction_amount_col='Amount'):

        self.customer_id_col = customer_id_col
        self.transaction_amount_col = transaction_amount_col
        self.agg_df_ = None # To store aggregated data during fit

    def fit(self, X, y=None):

        if not all(col in X.columns for col in [self.customer_id_col, self.transaction_amount_col]):
            raise ValueError(f"Required columns '{self.customer_id_col}' and '{self.transaction_amount_col}' "
                             f"not found in input DataFrame.")

        # Group by customer_id and calculate aggregate statistics
        self.agg_df_ = X.groupby(self.customer_id_col)[self.transaction_amount_col].agg(
            total_transaction_amount='sum',
            average_transaction_amount='mean',
            transaction_count='count',
            std_transaction_amount='std'
        ).reset_index()

        # Handle cases where std might be NaN for single transactions
        self.agg_df_['std_transaction_amount'] = self.agg_df_['std_transaction_amount'].fillna(0)

        return self

    def transform(self, X):

        if self.agg_df_ is None:
            raise NotFittedError("This AggregateFeaturesTransformer instance is not fitted yet. "
                                 "Call 'fit' before using this transformer.")

        # Merge the aggregate features back to the original DataFrame
        # Use left merge to keep all original rows and add aggregates
        X_transformed = X.merge(self.agg_df_, on=self.customer_id_col, how='left')

        for col in ['total_transaction_amount', 'average_transaction_amount',
                    'transaction_count', 'std_transaction_amount']:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].fillna(0) # Or a more appropriate default

        return X_transformed


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):

        self.datetime_col = datetime_col

    def fit(self, X, y=None):

        if self.datetime_col not in X.columns:
            raise ValueError(f"Datetime column '{self.datetime_col}' not found in input DataFrame.")
        return self

    def transform(self, X):

        X_transformed = X.copy()
        if self.datetime_col not in X_transformed.columns:
            raise ValueError(f"Datetime column '{self.datetime_col}' not found in input DataFrame during transform.")

        # Ensure the column is datetime type
        X_transformed[self.datetime_col] = pd.to_datetime(X_transformed[self.datetime_col], errors='coerce')

        # Extract features
        X_transformed['transaction_hour'] = X_transformed[self.datetime_col].dt.hour
        X_transformed['transaction_day'] = X_transformed[self.datetime_col].dt.day
        X_transformed['transaction_month'] = X_transformed[self.datetime_col].dt.month
        X_transformed['transaction_year'] = X_transformed[self.datetime_col].dt.year

        # Drop the original datetime column
        X_transformed = X_transformed.drop(columns=[self.datetime_col])

        return X_transformed


class CustomLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, encoder in self.encoders.items():
            try:
                X_transformed[col] = encoder.transform(X_transformed[col])
            except ValueError as e:
                warnings.warn(f"Unseen categories encountered in column '{col}' during Label Encoding transform. "
                              f"Error: {e}.")
                # Fallback: Convert to string, then encode, then convert unseen to a specific value
                unseen_mask = ~X_transformed[col].isin(encoder.classes_)
                X_transformed.loc[unseen_mask, col] = 'UNKNOWN_CATEGORY'
                encoder.classes_ = np.append(encoder.classes_, 'UNKNOWN_CATEGORY')
                X_transformed[col] = encoder.transform(X_transformed[col])

        return X_transformed

# --- Main Feature Engineering Pipeline ---
def build_feature_engineering_pipeline(
    numerical_cols_for_preprocessor, # Renamed for clarity
    categorical_onehot_cols,
    categorical_label_cols,
    datetime_col,
    customer_id_col,
    transaction_amount_col
):
    # 1. Pipeline for handling missing numerical values and scaling
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Impute missing numerical values with mean
        ('scaler', StandardScaler()) # Standardize numerical features
        # ('scaler', MinMaxScaler()) # Alternatively, Normalize to [0, 1]
    ])

    # 2. Pipeline for handling missing categorical values and One-Hot Encoding
    categorical_onehot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing categorical values with mode
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # One-Hot Encode
    ])

    # 3. Pipeline for Label Encoding (requires custom wrapper for ColumnTransformer)
    categorical_label_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing categorical values with mode
        ('label_encoder', CustomLabelEncoder()) # Label Encode
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols_for_preprocessor), # Use the renamed parameter
            ('cat_onehot', categorical_onehot_transformer, categorical_onehot_cols),
            ('cat_label', categorical_label_transformer, categorical_label_cols)
        ],
        remainder='passthrough' # Keep other columns not specified in transformers
    )
    feature_engineering_pipeline = Pipeline(steps=[
        ('aggregate_features', AggregateFeaturesTransformer(
            customer_id_col=customer_id_col,
            transaction_amount_col=transaction_amount_col
        )),
        ('datetime_extractor', DateTimeFeatureExtractor(
            datetime_col=datetime_col
        )),
        ('preprocessor', preprocessor)
    ])

    return feature_engineering_pipeline

if __name__ == "__main__":
    print("--- Starting Feature Engineering Pipeline Example ---")

    # 1. Load Raw Data (assuming 'data.csv' is available)
    print("\n1. Loading Raw Data...")
    df = pd.read_csv('data.csv') # Changed df_raw to df
   

    # Ensure 'TransactionStartTime' is datetime type
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce') # Changed df_raw to df

    print("Raw DataFrame Head:")
    print(df.head()) # Changed df_raw to df
    print("\nRaw DataFrame Info:")
    df.info() # 
    numerical_features_for_preprocessor = [
        'Amount', 'Value', 'CountryCode', 'PricingStrategy', 'FraudResult',
        # These are new features added by custom transformers, also numerical
        'total_transaction_amount', 'average_transaction_amount',
        'transaction_count', 'std_transaction_amount',
        'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year'
    ]

    # Categorical features to be One-Hot Encoded (low to medium cardinality)
    categorical_onehot_features = [
        'CurrencyCode',
        'ProductCategory'
    ]

    # Categorical features to be Label Encoded (if any, currently none based on image)
    categorical_label_features = []

    # Datetime column for extraction
    datetime_feature = 'TransactionStartTime'

    # Customer ID column for aggregation
    customer_id_feature = 'CustomerId'

    # Transaction amount column for aggregation
    transaction_amount_feature = 'Amount'

    # 2. Build the Feature Engineering Pipeline
    print("\n2. Building the Feature Engineering Pipeline...")
    fe_pipeline = build_feature_engineering_pipeline(
        numerical_cols_for_preprocessor=numerical_features_for_preprocessor, # Pass the comprehensive list
        categorical_onehot_cols=categorical_onehot_features,
        categorical_label_cols=categorical_label_features,
        datetime_col=datetime_feature,
        customer_id_col=customer_id_feature,
        transaction_amount_col=transaction_amount_feature
    )
    print("\n3. Fitting and Transforming the Data...")
    df_processed = fe_pipeline.fit_transform(df) # Changed df_raw to df
    preprocessor_step = fe_pipeline.named_steps['preprocessor']
    final_feature_names = preprocessor_step.get_feature_names_out()
    df_processed_final = pd.DataFrame(df_processed, columns=final_feature_names)


    print("\nProcessed DataFrame Head:")
    print(df_processed_final.head())
    print("\nProcessed DataFrame Info:")
    df_processed_final.info()
    print("\nProcessed DataFrame Description:")
    print(df_processed_final.describe())
    print("\n--- Verification ---")
    print("\nOriginal 'Amount' for customer C1:")
    print(df[df['CustomerId'] == 'C1']['Amount']) # Changed df_raw to df

    print("\nAggregate features for customer C1 (from raw data perspective):")
    cust1_raw_data = df[df['CustomerId'] == 'C1'] # Changed df_raw to df
    print(f"Total: {cust1_raw_data['Amount'].sum()}")
    print(f"Average: {cust1_raw_data['Amount'].mean()}")
    print(f"Count: {cust1_raw_data['Amount'].count()}")
    print(f"Std: {cust1_raw_data['Amount'].std()}")

    print("\nFirst few rows of processed data:")
    print(df_processed_final.iloc[:5])

    print("\nOriginal 'ProductCategory' and 'CurrencyCode' for first 5 rows:")
    print(df[['ProductCategory', 'CurrencyCode']].head()) # Changed df_raw to df

    print("\nOne-Hot Encoded 'ProductCategory' and 'CurrencyCode' for first 5 rows:")
    onehot_cols_in_final = [col for col in df_processed_final.columns if 'ProductCategory_' in col or 'CurrencyCode_' in col]
    print(df_processed_final[onehot_cols_in_final].head())

df_processed_final.to_csv('Final_feature_engineered_data.csv', index=False) 