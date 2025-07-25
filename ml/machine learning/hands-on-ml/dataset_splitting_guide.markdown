# Comprehensive Guide to Splitting Datasets in Machine Learning

This guide explains how to split a dataset into training and testing sets for machine learning, ensuring fair evaluation of model performance. It covers three methods: random splitting, identifier-based splitting, and stratified sampling. Each method is demonstrated with code examples, and best practices are provided to avoid common pitfalls like sampling bias.

---

## 1. Why Split a Dataset?

- **Purpose**: Split the dataset into a **training set** (to train the model) and a **test set** (to evaluate performance on unseen data).
- **Avoid Data Snooping Bias**: Looking at the test set too early can lead you to choose models or features based on patterns in the test data, causing overly optimistic performance estimates.
- **Goal**: Ensure the test set is representative of the real-world data the model will encounter.

---

## 2. Random Splitting with `train_test_split`

The simplest way to split a dataset is to randomly divide it into training and testing sets using Scikit-Learn’s `train_test_split` function.

### Code Example
```python
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

### Explanation
- **What it does**: Splits the dataset into 80% training and 20% testing (`test_size=0.2`).
- **Reproducibility**: `random_state=42` ensures the same split every time the code runs.
- **Output Example**: For a dataset with 20,640 instances, this yields approximately 16,512 training and 4,128 testing instances.

### Pros
- Quick and easy for large datasets.
- Works well when the dataset is large relative to the number of features.

### Cons
- **Inconsistent Splits**: Without `random_state`, each run produces a different test set, which can lead to the model “seeing” the entire dataset over time.
- **Sampling Bias**: In small datasets or when key features (e.g., income, gender) have uneven distributions, the test set may not represent the full dataset, leading to unreliable performance estimates.

### Example Issue
- If the U.S. population is 51.3% female and 48.7% male, random sampling might produce a test set with 60% female, skewing results. A survey company would avoid this by ensuring the sample matches the population’s gender ratio.

---

## 3. Consistent Splitting Using Identifiers

To ensure consistent test sets across multiple runs (even if the dataset is updated), you can use a unique identifier for each instance to determine whether it belongs to the training or test set.

### Code Example: Identifier-Based Splitting
```python
import hashlib
import numpy as np

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
```

### Using Row Index as Identifier
If the dataset lacks a unique identifier, use the row index. This assumes new data is appended to the end and no rows are deleted.

```python
# Use row index as ID
housing_with_id = housing_data.reset_index()  # Adds an 'index' column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
```

### Using Stable Features as Identifier
For more robustness, create an identifier from stable features, such as latitude and longitude in a housing dataset.

```python
# Create a unique ID from latitude and longitude
housing_with_id["id"] = housing_data["longitude"] * 1000 + housing_data["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
```

### Explanation
- **How it works**: Computes a hash of the identifier and uses the last byte to decide if an instance goes to the test set (e.g., if the hash value is < 256 * 0.2, it’s in the test set).
- **Benefit**: Ensures the same instances are selected for the test set across runs, even if the dataset is updated, as long as identifiers remain consistent.
- **Caution**: When using features like latitude and longitude, ensure they create unique IDs. Coarse location data may lead to duplicate IDs, introducing sampling bias.

---

## 4. Stratified Sampling for Important Features

Stratified sampling ensures the test set reflects the distribution of a key feature (e.g., income) in the full dataset. This is critical when certain features significantly impact model performance, reducing sampling bias.

### Step 1: Create Categories for Continuous Features
Convert continuous features (like `median_income`) into discrete categories to group similar values.

#### Code Example
```python
import numpy as np
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
```

- Divides `median_income` by 1.5 to create fewer categories (e.g., 1.0, 2.0, 3.0, 4.0, 5.0).
- Caps categories above 5 to avoid small, rare groups that could bias the split.
- Example: A `median_income` of 3.0 becomes category 2.0 (since `3.0 / 1.5 = 2.0`); a value of 7.5 becomes category 5.0.

### Step 2: Perform Stratified Sampling
Use `StratifiedShuffleSplit` to split the dataset while preserving the proportion of each category.

#### Code Example
```python
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

- Splits data into 80% training and 20% testing (`test_size=0.2`).
- Ensures the test set has the same income category proportions as the full dataset.
- `random_state=42` ensures reproducibility.

### Step 3: Verify Proportions
Check the distribution of categories in the full dataset to confirm the split is representative.

#### Code Example
```python
housing["income_cat"].value_counts() / len(housing)
```

**Example Output** (proportions):
- 3.0: 35.06%
- 2.0: 31.88%
- 4.0: 17.63%
- 5.0: 11.44%
- 1.0: 3.98%

The stratified test set (`strat_test_set`) will have nearly identical proportions, while a random split (from `train_test_split`) may be skewed.

### Step 4: Clean Up Temporary Columns
Remove temporary columns to restore the dataset.

#### Code Example
```python
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)
```

### Why Stratified Sampling?
- **Reduces Bias**: Ensures the test set mirrors the full dataset’s distribution for key features, like a well-designed survey (e.g., maintaining 51.3% female and 48.7% male).
- **Real-World Example**: Random sampling has a ~12% chance of producing a skewed test set (e.g., <49% or >54% female), which can bias results. Stratified sampling minimizes this risk.

---

## 5. Why It Matters

- **Avoid Biased Test Sets**: A skewed test set (e.g., too many high-income households) gives misleading performance metrics, overestimating or underestimating real-world performance.
- **Representative Sampling**: Stratified sampling ensures the test set reflects the population’s distribution, improving the reliability of model evaluation.
- **Cross-Validation**: These splitting techniques (especially stratified sampling) are also useful for cross-validation, which evaluates model performance more robustly.

---

## 6. Best Practices

- **Create Test Set Early**: Set aside the test set before exploring the data to avoid data snooping bias.
- **Use Random Splitting for Large Datasets**: `train_test_split` is sufficient when the dataset is large and features are evenly distributed.
- **Use Identifier-Based Splitting for Consistency**: Ensure consistent test sets across runs, especially when datasets are updated, by using stable identifiers (e.g., row index or computed IDs).
- **Use Stratified Sampling for Key Features**: Apply `StratifiedShuffleSplit` when features like income, gender, or age are critical to ensure a representative test set.
- **Ensure Reproducibility**: Set `random_state` or use identifier-based splitting for consistent results.
- **Clean Up Temporary Columns**: Remove helper columns (e.g., `income_cat`) after splitting to keep the dataset tidy.
- **Check for Sampling Bias**: For small datasets or uneven feature distributions, verify the test set’s representativeness (e.g., compare category proportions).

---

## 7. Visualizing the Benefit

- A histogram comparing income categories in the full dataset, stratified test set, and random test set shows that stratified sampling closely matches the full dataset’s distribution, while random sampling often introduces skew.
- This ensures the test set is a reliable proxy for real-world data, leading to more trustworthy model evaluations.

---

## 8. Additional Notes

- **When to Use Stratified Sampling**: Essential for small datasets or when a feature (e.g., income) is critical to the model’s performance.
- **Dataset Updates**: Identifier-based splitting is robust to dataset updates, unlike random splitting with `train_test_split`, which may break if new data is added.
- **Caution with Feature-Based IDs**: Ensure feature-based IDs (e.g., latitude and longitude) are unique to avoid unintended duplicates, which could introduce bias.
- **Cross-Validation**: The concepts here (e.g., stratified sampling) are foundational for cross-validation, which splits the data multiple times to assess model stability.

By carefully choosing the right splitting method, you ensure your model is evaluated on a representative test set, leading to more reliable performance estimates in real-world scenarios.