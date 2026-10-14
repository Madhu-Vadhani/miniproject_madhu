
import pandas as pd

file_path = "data/raw/online_retail_II.xlsx"

print("Loading dataset...")
sheets = pd.read_excel(file_path, sheet_name=None)
df = pd.concat(sheets.values(), ignore_index=True)
df.columns = df.columns.str.strip()

# Convert dates and remove rows without customer IDs
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df = df.dropna(subset=["Customer ID", "InvoiceDate"])

# Remove cancellations, returns, and non-positive transactions
df = df[~df["Invoice"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

# Keep product-like stock codes: exclude codes containing letters
df["StockCode"] = df["StockCode"].astype(str).str.strip()
df = df[df["StockCode"].str.fullmatch(r"\d+")]

# Sort chronologically
df = df.sort_values("InvoiceDate")

# Time-based split: final 20% of dates for testing
split_date = df["InvoiceDate"].max() - pd.DateOffset(months=2)

train = df[df["InvoiceDate"] < split_date].copy()
test = df[df["InvoiceDate"] >= split_date].copy()

print("\nCleaning complete!")
print("Cleaned rows:", f"{len(df):,}")
print("Training rows:", f"{len(train):,}")
print("Testing rows:", f"{len(test):,}")
print("Split date:", split_date)
print("Training date range:", train["InvoiceDate"].min(), "to", train["InvoiceDate"].max())
print("Testing date range:", test["InvoiceDate"].min(), "to", test["InvoiceDate"].max())
print("Training customers:", train["Customer ID"].nunique())
print("Training products:", train["StockCode"].nunique())



import numpy as np
from scipy.sparse import coo_matrix, save_npz

# 1. Aggregate repeated purchases by each customer for each product
interactions = (
    train.groupby(["Customer ID", "StockCode"])["Quantity"]
    .sum()
    .reset_index()
)

# 2. Keep products purchased by at least 10 distinct customers
product_customer_counts = interactions.groupby("StockCode")["Customer ID"].nunique()
popular_products = product_customer_counts[
    product_customer_counts >= 10
].index

interactions = interactions[
    interactions["StockCode"].isin(popular_products)
].copy()

# 3. Convert purchase quantity to log(1 + quantity)
interactions["strength"] = np.log1p(interactions["Quantity"])

# 4. Map customers and products to matrix positions
customers = sorted(interactions["Customer ID"].unique())
products = sorted(interactions["StockCode"].unique())

customer_to_idx = {customer: i for i, customer in enumerate(customers)}
product_to_idx = {product: i for i, product in enumerate(products)}

rows = interactions["Customer ID"].map(customer_to_idx).to_numpy()
cols = interactions["StockCode"].map(product_to_idx).to_numpy()
values = interactions["strength"].to_numpy(dtype=np.float32)

# 5. Build a sparse customer-product matrix
A = coo_matrix(
    (values, (rows, cols)),
    shape=(len(customers), len(products)),
    dtype=np.float32
).tocsr()

# 6. Report matrix size
dense_size_mb = len(customers) * len(products) * 8 / (1024 ** 2)
sparse_size_mb = (
    A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
) / (1024 ** 2)

print("\n--- MATRIX REPORT ---")
print("Matrix shape:", A.shape)
print("Customers:", len(customers))
print("Products:", len(products))
print("Non-zero interactions:", A.nnz)
print("Matrix density:", round(A.nnz / (A.shape[0] * A.shape[1]) * 100, 3), "%")
print("Dense matrix size (float64):", round(dense_size_mb, 2), "MB")
print("Sparse matrix storage (float32):", round(sparse_size_mb, 2), "MB")

# Save the sparse matrix for later analysis
save_npz("customer_product_matrix.npz", A)
print("Saved customer_product_matrix.npz")


from sklearn.decomposition import TruncatedSVD
import time

print("\n--- RECOMMENDATION EVALUATION ---")

# Build the held-out test purchases using training customer/product mappings
test_eval = test[
    test["Customer ID"].isin(customer_to_idx) &
    test["StockCode"].isin(product_to_idx)
].copy()

test_eval["customer_idx"] = test_eval["Customer ID"].map(customer_to_idx)
test_eval["product_idx"] = test_eval["StockCode"].map(product_to_idx)

# Relevant test items must be products not already bought in training
train_seen = interactions.groupby("Customer ID")["StockCode"].apply(set).to_dict()

test_truth = {}
for customer, group in test_eval.groupby("Customer ID"):
    seen = train_seen.get(customer, set())
    relevant = set(group["StockCode"]) - seen
    relevant = {p for p in relevant if p in product_to_idx}
    if relevant:
        test_truth[customer_to_idx[customer]] = {
            product_to_idx[p] for p in relevant
        }

print("Customers evaluated:", len(test_truth))

# Fit Truncated SVD using training matrix only
start = time.time()
svd = TruncatedSVD(n_components=50, random_state=42)
customer_factors = svd.fit_transform(A)
scores = customer_factors @ svd.components_
svd_seconds = time.time() - start

# Rank products for each customer, excluding products already purchased
for customer_idx, customer in enumerate(customers):
    seen = train_seen.get(customer, set())
    for product in seen:
        p_idx = product_to_idx.get(product)
        if p_idx is not None:
            scores[customer_idx, p_idx] = -np.inf

# Top 10 products by training interaction count (baseline)
popular = (
    interactions.groupby("StockCode")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .index.tolist()
)
popular_idx = [
    product_to_idx[p] for p in popular if p in product_to_idx
]

def precision_at_10(recommendations, truth):
    if not truth:
        return 0
    return len(set(recommendations[:10]) & truth) / 10

svd_scores = []
baseline_scores = []

for customer_idx, relevant in test_truth.items():
    svd_top10 = np.argsort(scores[customer_idx])[-10:][::-1]
    baseline_top10 = [
        p for p in popular_idx
        if p not in {
            product_to_idx[x] for x in train_seen.get(customers[customer_idx], set())
            if x in product_to_idx
        }
    ][:10]

    svd_scores.append(precision_at_10(svd_top10, relevant))
    baseline_scores.append(precision_at_10(baseline_top10, relevant))

print(f"SVD fit time: {svd_seconds:.2f} seconds")
print(f"SVD Precision@10: {np.mean(svd_scores):.4f}")
print(f"Best-seller Precision@10: {np.mean(baseline_scores):.4f}")
print(f"SVD explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}")


# --- CUSTOMER SEGMENTATION (ML6) ---

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print("\n--- CUSTOMER SEGMENTATION ---")

# Cluster customers using their 50-dimensional SVD representations
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(customer_factors)

# Map each customer to a cluster
customer_clusters = pd.DataFrame({
    "Customer ID": customers,
    "cluster": cluster_labels
})

# Plot customers using their first two SVD dimensions
plt.figure(figsize=(10, 6))
for cluster_id in range(4):
    mask = cluster_labels == cluster_id
    plt.scatter(
        customer_factors[mask, 0],
        customer_factors[mask, 1],
        label=f"Segment {cluster_id + 1}",
        alpha=0.6,
        s=12
    )

plt.title("Customer Segments Based on Purchasing Patterns")
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.legend()
plt.tight_layout()
plt.savefig("customer_segments.png", dpi=200)
plt.close()

# Describe each segment using its most purchased products
profile_data = interactions.merge(
    customer_clusters, on="Customer ID"
)

for cluster_id in range(4):
    segment = profile_data[
        profile_data["cluster"] == cluster_id
    ]

    top_products = (
        segment.groupby("StockCode")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    segment_customers = customer_clusters.loc[
        customer_clusters["cluster"] == cluster_id,
        "Customer ID"
    ]

    top_countries = (
        train[train["Customer ID"].isin(segment_customers)]
        ["Country"].value_counts().head(3).index.tolist()
    )

    print(f"\nSegment {cluster_id + 1}")
    print("Top products:", top_products)
    print("Top countries:", top_countries)
    print("Number of customers:", len(segment_customers))

print("\nSaved customer_segments.png")
