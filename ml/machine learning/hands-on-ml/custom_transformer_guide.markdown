# Creating Custom Transformers for Machine Learning

This guide explains how to create a custom transformer in Scikit-Learn to perform specific data preparation tasks, like adding new features. It uses a simple example from the housing dataset and shows how to make the transformer work with Scikit-Learn’s pipelines and hyperparameter tuning.

---

## Why Create a Custom Transformer?

- **Custom Tasks**: Perform operations not covered by Scikit-Learn, like creating new features (e.g., rooms per household).
- **Seamless Integration**: Make your transformer work with Scikit-Learn’s tools, like pipelines and grid search.
- **Flexibility**: Use hyperparameters to control which transformations to apply, allowing easy experimentation.

---

## How to Create a Custom Transformer

To make a transformer compatible with Scikit-Learn:
1. Create a class with three methods: `fit()`, `transform()`, and `fit_transform()`.
2. Inherit from `TransformerMixin` to get `fit_transform()` for free.
3. Inherit from `BaseEstimator` to get `get_params()` and `set_params()` for hyperparameter tuning.
4. Avoid `*args` and `**kwargs` in the constructor to ensure compatibility with `BaseEstimator`.

### Example: Adding Combined Features
This transformer adds two new features to the housing dataset:
- **Rooms per household**: Total rooms ÷ Households
- **Population per household**: Population ÷ Households
- **Bedrooms per room** (optional): Total bedrooms ÷ Total rooms

It includes a hyperparameter to control whether to add `bedrooms_per_room`.

#### Code Example
```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Column indices for the housing dataset (adjust based on your data)
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # Hyperparameter
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self  # Nothing to compute/learn
    
    def transform(self, X, y=None):
        # Calculate new features
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        
        # Conditionally add bedrooms_per_room
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Usage
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

---

## How It Works

- **Class Structure**:
  - Inherits `BaseEstimator` (for hyperparameter tuning) and `TransformerMixin` (for `fit_transform()`).
  - `__init__`: Sets a hyperparameter `add_bedrooms_per_room` (default: `True`) to decide whether to include `bedrooms_per_room`.
  
- **Methods**:
  - `fit()`: Returns `self` (no learning needed, as we’re just computing ratios).
  - `transform()`: Adds new features by dividing columns and stacks them with the original data using `np.c_`.
  - `fit_transform()`: Provided by `TransformerMixin` (combines `fit()` and `transform()`).

- **Hyperparameter**:
  - `add_bedrooms_per_room` lets you test if adding `bedrooms_per_room` improves your model.
  - Use `get_params()` and `set_params()` (from `BaseEstimator`) for automatic tuning with tools like `GridSearchCV`.

- **Input/Output**:
  - Input: A NumPy array (e.g., `housing.values`) with columns for rooms, bedrooms, population, and households.
  - Output: A NumPy array with the original columns plus the new features.

---

## Why Use Hyperparameters?

- **Experimentation**: Toggle features (e.g., `bedrooms_per_room`) to see if they help the model.
- **Automation**: Hyperparameters allow tools like `GridSearchCV` to test different combinations automatically.
- **Time-Saving**: Automating data preparation steps lets you try many feature combinations quickly.

---

## Key Takeaways

- **Custom Transformers**: Create your own transformers for tasks like adding new features or custom cleaning.
- **Scikit-Learn Compatibility**: Use `fit()`, `transform()`, and inherit `TransformerMixin` and `BaseEstimator` for seamless integration.
- **Hyperparameters**: Add flexibility to control which transformations to apply, enabling experimentation.
- **Use in Pipelines**: Combine with other Scikit-Learn tools (e.g., `Imputer`, `OneHotEncoder`) in a pipeline for streamlined preprocessing.

---

## Notes

- **Column Indices**: Adjust `rooms_ix`, `bedrooms_ix`, etc., based on your dataset’s column order.
- **Data Type**: The transformer expects a NumPy array. Convert DataFrames to arrays using `housing.values`.
- **Pipelines**: This transformer can be used in a Scikit-Learn `Pipeline` for automated preprocessing.
- **Testing Features**: Use the hyperparameter to test whether new features (like `bedrooms_per_room`) improve model performance.

This transformer makes it easy to add custom features to your dataset while keeping your code compatible with Scikit-Learn’s ecosystem.