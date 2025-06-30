import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# --- 1. Generate Sample Transaction Data ---
# In a real scenario, you would load your transaction data here.
df_transactions = pd.read_csv('./Final_preprocessed_data.csv')

np.random.seed(42) # for reproducibility of synthetic data

num_customers = 500
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)

data = []
for customer_id in range(1, num_customers + 1):
    # Simulate varying transaction frequencies and amounts
    num_transactions = np.random.randint(1, 30) # Customers can have 1 to 30 transactions
    for _ in range(num_transactions):
        random_days = np.random.randint(0, (end_date - start_date).days)
        transaction_date = start_date + timedelta(days=random_days)
        amount = np.random.uniform(10, 1000) # Transaction amounts between 10 and 1000
        data.append({'CustomerId': customer_id, 'TransactionDate': transaction_date, 'Amount': amount})

df_transactions = pd.DataFrame(data)

print("--- Sample Transaction Data ---")
print(df_transactions.head())
print(f"\nTotal transactions: {len(df_transactions)}")
print(f"Unique customers: {df_transactions['CustomerId'].nunique()}")

# --- 2. Calculate RFM Metrics ---

# Define a snapshot date: The day after the latest transaction in the dataset
snapshot_date = df_transactions['TransactionDate'].max() + timedelta(days=1)
print(f"\nSnapshot Date for Recency calculation: {snapshot_date}")

# Calculate RFM for each customer
rfm_df = df_transactions.groupby('CustomerId').agg(
    Recency=('TransactionDate', lambda date: (snapshot_date - date.max()).days),
    Frequency=('TransactionDate', 'count'),
    Monetary=('Amount', 'sum')
).reset_index()

print("\n--- Calculated RFM Metrics (Raw) ---")
print(rfm_df.head())
print(f"\nRFM DataFrame shape: {rfm_df.shape}")


# Initialize StandardScaler
scaler = StandardScaler()

# Select RFM features for scaling
rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]

# Scale the features
scaled_rfm_features = scaler.fit_transform(rfm_features)
scaled_rfm_df = pd.DataFrame(scaled_rfm_features, columns=rfm_features.columns, index=rfm_df.index)

print("\n--- Scaled RFM Features ---")
print(scaled_rfm_df.head())

# --- 4. Cluster Customers using K-Means ---

# Set random_state for reproducibility
random_state = 42
kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10) # n_init for robust centroid initialization

# Fit K-Means to the scaled data
kmeans.fit(scaled_rfm_df)

# Add cluster labels to the RFM DataFrame
rfm_df['Cluster'] = kmeans.labels_

print("\n--- RFM Data with Cluster Labels ---")
print(rfm_df.head())
print(f"\nCluster distribution:\n{rfm_df['Cluster'].value_counts()}")

cluster_analysis = rfm_df.groupby('Cluster').agg(
    AvgRecency=('Recency', 'mean'),
    AvgFrequency=('Frequency', 'mean'),
    AvgMonetary=('Monetary', 'mean'),
    Count=('CustomerId', 'count')
).sort_values(by=['AvgRecency', 'AvgFrequency', 'AvgMonetary'], ascending=[False, True, True]) # Sort to find high-risk

print("\n--- Cluster Analysis (Mean RFM Values) ---")
print(cluster_analysis)
high_risk_cluster_id = cluster_analysis.index[0]
print(f"\nIdentified High-Risk Cluster ID: {high_risk_cluster_id}")

# Create the new binary target column 'is_high_risk'
rfm_df['is_high_risk'] = rfm_df['Cluster'].apply(lambda x: 1 if x == high_risk_cluster_id else 0)

print("\n--- RFM Data with 'is_high_risk' Label ---")
print(rfm_df.head())
print(f"\nHigh-risk customer count: {rfm_df['is_high_risk'].sum()}")

# --- 6. Integrate the Target Variable ---

main_data = []
for customer_id in range(1, num_customers + 1):
    age = np.random.randint(20, 70)
    income = np.random.uniform(30000, 100000)
    main_data.append({'CustomerId': customer_id, 'Age': age, 'Income': income})

df_main = pd.DataFrame(main_data)

print("\n--- Sample Main Processed Dataset (Before Merge) ---")
print(df_main.head())
print(f"\nMain dataset shape: {df_main.shape}")

# Merge the 'is_high_risk' column back into the main processed dataset
# We only need CustomerId and is_high_risk from rfm_df
df_main = pd.merge(df_main, rfm_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

print("\n--- Main Processed Dataset (After Merge with 'is_high_risk') ---")
print(df_main.head())
print(f"\nMain dataset shape after merge: {df_main.shape}")

# Verify that all customers from the main dataset have an 'is_high_risk' label
print(f"\nMissing 'is_high_risk' values: {df_main['is_high_risk'].isnull().sum()}")

# Final check: distribution of the new target variable
print(f"\nDistribution of 'is_high_risk':\n{df_main['is_high_risk'].value_counts()}")
