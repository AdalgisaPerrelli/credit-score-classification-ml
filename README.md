# Credit Score Classification with Machine Learning

This project analyzes a Kaggle credit score dataset to predict each customer’s credit score class (`Poor`, `Standard`, `Good`) using machine learning methods for clustering and classification.

---

## Project overview

- Dataset: 100,000 customers, 28 variables (demographics, banking profile, credit usage and repayment behaviour).  
- Target: `CREDIT_SCORE` with three classes (`Poor`, `Standard`, `Good`).  
- Goal: build and compare models to (i) explore natural groupings in the data, (ii) predict the credit score of new customers.

---

## Data preparation

- Removed redundant identifiers (`ID`, `SSN`, `NAME`) and problematic variables (`TYPE_LOAN`, `CREDIT_AGE`).  
- Corrected variable types and handled anomalies at record level.  
- Imputed missing values using a mix of rule‑based strategies (mode within `CUSTOMER_ID`) and a non‑parametric multiple imputation method.  
- Detected and removed rows containing strong outliers.  
- Performed correlation analysis to drop highly collinear variables (e.g. `MONTHLY_SALARY`).  
- Collapsed the panel into one record per customer, obtaining a final dataset of 11,257 rows and 22 variables.  
- Selected the most important features via Random Forest variable importance.

---

## Modelling

### Unsupervised learning

- Prepared data by standardizing numerical features and encoding categorical variables (e.g. dummy variables for `CREDIT_MIX`).  
- Applied several clustering techniques:  
  - K‑Means / K‑Medians (Euclidean and Manhattan distances, choice of \(k\) via Elbow and Silhouette).  
  - Hierarchical agglomerative clustering with different linkages and distance metrics.  
  - DBSCAN for density‑based clustering and outlier detection.  
- Clusters were evaluated against the true `CREDIT_SCORE` labels using cluster purity and silhouette indices; only moderate separation was observed, especially due to the similarity of the `Standard` class with the other two.

### Supervised learning

- Split data into training and test sets; tuned models via cross‑validation.  
- Implemented and compared multiple classifiers:  
  - k‑Nearest Neighbours (different values of \(k\), moderate overall accuracy).  
  - Support Vector Machines with linear and radial kernels (radial SVM achieved the best overall performance, with ~70% test accuracy and balanced class‑wise metrics).  
  - Random Forest (hyperparameters tuned via repeated cross‑validation, out‑of‑bag error around 30%).  
  - Neural network classifier (used as an additional benchmark).  
- All models confirm that the `Standard` class is hardest to separate cleanly from `Poor` and `Good`.

---

## Key results

- Clustering methods do not naturally recover the three credit score classes with high purity.  
- Supervised models achieve substantially better performance; the radial SVM provides the best trade‑off between overall accuracy and class‑wise sensitivity/specificity.  
- Predicting the `Standard` class remains challenging across models, highlighting intrinsic overlap in customer profiles.

---

## Repository structure (suggested)

