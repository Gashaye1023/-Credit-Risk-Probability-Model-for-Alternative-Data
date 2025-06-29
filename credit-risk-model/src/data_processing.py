import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# --- 1. Custom Transformers for Feature Creation ---
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='CustomerId', agg_col='Amount'):
        self.group_col = group_col
        self.agg_col = agg_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.group_col not in X.columns or self.agg_col not in X.columns:
            raise ValueError(f"Required columns '{self.group_col}' or '{self.agg_col}' not found in DataFrame.")

        # Calculate aggregate features
        agg_data = X.groupby(self.group_col)[self.agg_col].agg(
            total_transaction_amount='sum',
            average_transaction_amount='mean',
            transaction_count='count',
            std_transaction_amount='std'
        ).reset_index()

        # Handle cases where std might be NaN for single transactions
        agg_data['std_transaction_amount'] = agg_data['std_transaction_amount'].fillna(0)
        unique_customers_with_country = X[[self.group_col, 'CountryCode']].drop_duplicates()
        X_transformed = pd.merge(unique_customers_with_country, agg_data, on=self.group_col, how='left')
        
        return X_transformed


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.datetime_col not in X_copy.columns:
            raise ValueError(f"Datetime column '{self.datetime_col}' not found in DataFrame.")

        # Ensure the column is datetime type
        X_copy[self.datetime_col] = pd.to_datetime(X_copy[self.datetime_col], errors='coerce')

        # Extract features
        X_copy['transaction_hour'] = X_copy[self.datetime_col].dt.hour
        X_copy['transaction_day'] = X_copy[self.datetime_col].dt.day
        X_copy['transaction_month'] = X_copy[self.datetime_col].dt.month
        X_copy['transaction_year'] = X_copy[self.datetime_col].dt.year

        return X_copy


# --- 2. Function to create the Feature Engineering Pipeline ---

def create_feature_engineering_pipeline(
    numerical_features,
    categorical_onehot_features,
    categorical_label_features,
    imputation_strategy_numerical='mean',
    imputation_strategy_categorical='most_frequent',
    scaler_type='standard' # 'standard' or 'minmax'
):

    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputation_strategy_numerical)),
        ('scaler', StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
    ])

    # Preprocessing for categorical features (One-Hot Encoding)
    onehot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputation_strategy_categorical)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat_onehot', onehot_transformer, categorical_onehot_features)
        ],
        remainder='drop' # <--- CHANGED THIS FROM 'passthrough' TO 'drop'
    )

    # Full pipeline
    pipeline = Pipeline(steps=[
        # Step 1: Extract time-based features (applied to the raw data first)
        ('time_extractor', TimeFeatureExtractor(datetime_col='TransactionStartTime')),
        # Step 2: Apply aggregate features (This transformer will now also carry 'CountryCode')
        ('agg_features', AggregateFeatures(group_col='CustomerId', agg_col='Amount')),
        # Step 3: Apply numerical scaling and categorical encoding
        ('preprocessor', preprocessor)
    ])
    
    return pipeline


if __name__ == '__main__':
    df = pd.read_csv('data/raw/data.csv')
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")

    numerical_features_final = [
        'total_transaction_amount',
        'average_transaction_amount',
        'transaction_count',
        'std_transaction_amount',
        'CountryCode' 
    ]     
    categorical_onehot_features_final = [
        # 'ProductCategory', # Example if aggregated to customer level (e.g., most frequent)
        # 'PaymentMethod' # Example from original dummy data, if applicable
    ]
    
    # Label encoding is typically applied to target variable or specific columns.
    categorical_label_features_final = [] 
    # Create the full pipeline
    full_pipeline = create_feature_engineering_pipeline(
        numerical_features=numerical_features_final,
        categorical_onehot_features=categorical_onehot_features_final,
        categorical_label_features=categorical_label_features_final,
        imputation_strategy_numerical='mean',
        imputation_strategy_categorical='most_frequent',
        scaler_type='standard'
    )

    # Fit and transform the data using the full pipeline
    transformed_data = full_pipeline.fit_transform(df)

    print("Transformed Data (Numpy Array after full pipeline):")
    print(transformed_data)
    print("\nShape of transformed data:", transformed_data.shape)
    scaled_feature_names = numerical_features_final
    
    transformed_df = pd.DataFrame(transformed_data, columns=scaled_feature_names)
    print("\nTransformed Data (Pandas DataFrame):")
    print(transformed_df)
    print("\n" + "="*50 + "\n")
    print("Example of Label Encoding for 'ProductCategory' (if needed for target or specific feature):")
    le = LabelEncoder()
    if 'ProductCategory' in df.columns:
        df['ProductCategory_encoded'] = le.fit_transform(df['ProductCategory'])
        print(df[['ProductCategory', 'ProductCategory_encoded']].drop_duplicates())
    else:
        print("'ProductCategory' column not found in the original DataFrame for Label Encoding example.")
    print("\n" + "="*50 + "\n")

    print("Example of removing rows with missing 'Amount' (if few):")
    df_cleaned_by_removal = df.dropna(subset=['Amount'])
    print(df_cleaned_by_removal[['CustomerId', 'Amount']])
    print("\nOriginal DataFrame shape:", df.shape)
    print("DataFrame shape after removal:", df_cleaned_by_removal.shape)
    print("\nTransformed DataFrame head:",transformed_df.head())
    print("\nTransformed DataFrame shape:",transformed_df.shape)
    df_cleaned_by_removal.to_csv('data/processed/cleaned_data.csv', index=False)
    transformed_df.to_csv('data/processed/transformed_data.csv', index=False)
