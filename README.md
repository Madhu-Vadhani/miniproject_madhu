# Customers Like You: Retail Product Recommendation Using SVD

## Project Overview
This project explores retail purchase history to build a customer-product interaction matrix, generate personalized product recommendations using Truncated Singular Value Decomposition (SVD), and identify customer segments based on purchasing patterns.

## Dataset
- **Source:** UCI Machine Learning Repository, Online Retail II
- **Input file:** `data/raw/online_retail_II.xlsx`
- The workbook contains two sheets that are loaded and combined by the pipeline.
- The raw dataset is not included in the Git repository.

## Project Goals
1. Clean and profile the retail transaction data.
2. Build a sparse customer-product matrix using training transactions only.
3. Apply Truncated SVD to learn lower-dimensional customer and product representations.
4. Compare personalized recommendations against a bestseller baseline.
5. Cluster customers and visualize the resulting segments.
6. Compare SVD with PCA and evaluate recommendations on held-out purchases.

## Data Processing Decisions
- Remove rows without a customer ID or invoice date.
- Remove cancellation invoices, non-positive quantities, and non-positive prices.
- Exclude non-product stock codes that do not contain digits only.
- Aggregate purchase quantities by customer and product.
- Apply `log(1 + quantity)` to reduce the influence of very large quantities.
- Retain products purchased by at least 10 distinct customers.
- Store the customer-product matrix in sparse CSR format to reduce memory usage.
- Use a fixed random seed of 42 for reproducible model steps.

**Note:** The time split and evaluation metrics must be checked against the assignment requirements before final submission.

## Initial Results
The current implementation reported:
- Sparse matrix shape: 5,253 customers × 2,801 products
- Non-zero interactions: 347,231
- Matrix density: 2.36%
- Sparse matrix storage: approximately 2.67 MB
- Truncated SVD components: 50
- Initial SVD Precision@10: 0.0660
- Initial bestseller Precision@10: 0.0224
- Initial SVD fit time: approximately 0.25 seconds

These are preliminary results from the current implementation and may change after correcting the time split and finalizing the evaluation.

## Requirements
Python 3.11 or a compatible Python version. Install the packages listed in `requirements.txt`.

## Setup
1. Clone this repository.
2. Place the original dataset at `data/raw/online_retail_II.xlsx`.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the current pipeline:

   ```bash
   python main.py
   ```

The script loads and cleans the dataset, builds the sparse matrix, evaluates recommendations, and creates `customer_segments.png`.

## Output Files
- `customer_product_matrix.npz`: sparse customer-product interaction matrix
- `customer_segments.png`: customer segmentation visualization

## Limitations and Next Steps
- Verify the required final-two-month test split.
- Calculate Hit Rate@10 in addition to Precision@10.
- Compare PCA and Truncated SVD using the same matrix and component count.
- Evaluate multiple component counts and document model selection.
- Improve segment descriptions using product and