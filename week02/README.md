# Week 2: Decision Trees and Random Forests

This project extends Week 1's house price analysis by investigating **Decision Trees** and **Random Forests** for regression and classification.

The focus is on understanding how tree-based models work, how hyperparameters affect model performance, how overfitting can be identified and controlled, and whether ensemble learning improves predictions compared with a single Decision Tree.

The analysis builds on the dataset, cleaning procedures, EDA and baseline models established in Week 1. For details about these steps, refer to the [Week 1 project](../Week-01/README.md).

---

## Project overview

The project investigates two machine learning tasks:

1. **House price prediction** using Decision Tree Regression and Random Forest Regression
2. **High-price classification** using Decision Tree Classification and Random Forest Classification

The main workflow is:

**Data preparation → Model training → Hyperparameter experiments → Evaluation → Model comparison → Feature importance**

The original dataset was cleaned again using the Week 1 preprocessing procedure. The same 80/20 train/test split and `random_state=42` were retained to support comparison with the baseline models.

For classification, `sale_price_eur` was excluded from the input features to prevent data leakage because the target `high_price` is derived directly from the sale price.

---

## Part A — Decision Tree Regression

A `DecisionTreeRegressor` was trained to predict `sale_price_eur`. Several configurations were evaluated to investigate how tree complexity affects predictive performance.

The experiments considered:

- `max_depth` — maximum depth of the tree
- `min_samples_split` — minimum number of samples required to split a node
- `min_samples_leaf` — minimum number of samples required in a leaf

### Results

| Configuration | MAE (EUR) | RMSE (EUR) | R² |
|---|---:|---:|---:|
| Shallow tree | 164,646 | 229,826 | 0.678 |
| Medium tree | 172,903 | 242,878 | 0.641 |
| Unrestricted tree | 186,233 | 258,641 | 0.592 |

The **shallow tree** achieved the best test performance among the tested Decision Tree configurations.

The unrestricted tree achieved nearly perfect training performance, with a training RMSE of approximately zero and an R² of 1.000. However, its test RMSE increased to approximately €258,641.

This difference provides clear evidence of **overfitting**: the tree fitted the training observations extremely well but did not generalise as effectively to unseen data.

The best tested configuration was:

```python
DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
````

## Part B — Random Forest Regression

A `RandomForestRegressor` was trained to improve upon the performance of a single Decision Tree by combining predictions from multiple trees.

The experiments investigated the number of trees, maximum depth, minimum leaf size, and number of features considered at each split.

### Results

| Configuration | MAE (EUR) | RMSE (EUR) | R² |
| --- | --- | --- | --- |
| 100 trees | 134,890 | 183,364 | 0.795 |
| 300 trees | 135,303 | 182,791 | 0.796 |
| Tuned Random Forest | 133,729 | 182,442 | 0.797 |

The tuned Random Forest achieved the lowest MAE and RMSE and the highest R² among the tested configurations.

The best tested configuration was:

```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    max_features=0.7,
    random_state=42
)
```

Compared with the best tested Decision Tree, the Random Forest reduced test RMSE from approximately €229,826 to €182,442.

This improvement illustrates how combining multiple trees can produce more stable predictions than relying on a single tree.

## Part C — Regression model comparison

The best Week 2 regression models were compared with the Week 1 Linear Regression baseline.

| Model | MAE (EUR) | RMSE (EUR) | R² |
| --- | --- | --- | --- |
| Linear Regression | 100,874 | 140,585 | 0.880 |
| Decision Tree | 164,646 | 229,826 | 0.678 |
| Random Forest | 133,729 | 182,442 | 0.797 |

Linear Regression performed best overall on the selected test split, achieving the lowest prediction errors and the highest R².

Random Forest Regression substantially outperformed the single Decision Tree but did not surpass the Linear Regression baseline.

This demonstrates that a more flexible model does not necessarily produce better predictions. Model selection should be based on measured performance rather than complexity alone.

## Part D — Decision Tree classification

A `DecisionTreeClassifier` was trained to predict `high_price` using property characteristics rather than the sale price itself.

Maximum depths of 3, 5, 8, and unrestricted depth were tested.

### Results

| Max. depth | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| 3 | 85.0% | 91.4% | 84.2% | 87.7% |
| 5 | 86.3% | 91.6% | 86.2% | 88.8% |
| 8 | 84.2% | 91.9% | 82.2% | 86.8% |
| Unrestricted | 84.6% | 90.8% | 84.2% | 87.4% |

The depth-5 configuration achieved the highest test accuracy and F1 Score among the tested Decision Trees.

The results suggest that controlling tree depth can improve generalisation, while allowing a tree to become more complex does not guarantee better classification performance.

The best tested configuration was:

```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=2,
    random_state=42
)
```

## Part E — Random Forest classification

A `RandomForestClassifier` was trained to improve high-price classification by combining multiple Decision Trees.

### Results

| Configuration | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| 100 trees | 90.8% | 93.3% | 92.1% | 92.7% |
| 300 trees | 90.4% | 92.7% | 92.1% | 92.4% |
| Tuned Random Forest | 90.8% | 92.2% | 93.4% | 92.8% |

The tuned Random Forest achieved the highest F1 Score among the tested Random Forest configurations.

The best tested configuration was:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    max_features=0.7,
    random_state=42
)
```

The model achieved an F1 Score of approximately 92.8%, demonstrating a substantial improvement over the single Decision Tree classifier.

## Part F — Classification model comparison

The best classification models from Week 2 were compared with Week 1's Logistic Regression baseline.

| Model | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| Logistic Regression | 92.9% | 94.1% | 94.7% | 94.4% |
| Decision Tree | 86.3% | 91.6% | 86.2% | 88.8% |
| Random Forest | 90.8% | 92.2% | 93.4% | 92.8% |

Logistic Regression achieved the strongest overall classification performance on the selected test split.

Random Forest improved substantially over the single Decision Tree, although it did not outperform the Logistic Regression baseline.

The results show that ensemble learning can improve upon an individual tree without outperforming a simpler model.

## Part G — Random Forest feature importance

Feature importance was extracted from both Random Forest models to investigate which property characteristics contributed most to their predictions.

### Regression: Top features

| Feature | Importance |
| --- | --- |
| Living area | 0.449 |
| Lot size | 0.156 |
| Munich | 0.134 |
| Leipzig | 0.038 |
| Bedrooms | 0.027 |
| Distance to city centre | 0.026 |
| Year built | 0.019 |
| Crime rate | 0.019 |
| School rating | 0.018 |
| Maxvorstadt | 0.009 |

Living area was the most important feature for house-price prediction, followed by lot size and the Munich location indicator.

### Classification: Top features

| Feature | Importance |
| --- | --- |
| Lot size | 0.325 |
| Living area | 0.274 |
| Munich | 0.071 |
| Crime rate | 0.037 |
| Year built | 0.036 |
| Distance to city centre | 0.032 |
| School rating | 0.027 |
| Bedrooms | 0.023 |
| Leipzig | 0.019 |
| Westend | 0.018 |

For high-price classification, lot size and living area were the two most important features.

Location indicators also contributed to both models, suggesting that geographical differences provide useful predictive information.

Feature importance represents the contribution of features to the model's predictions. It does not establish a causal relationship between a property characteristic and its sale price.

## Key findings

The Week 2 experiments produced several important findings:

- Decision Tree overfitting: An unrestricted regression tree fitted the training data almost perfectly but performed substantially worse on the test set.
- Hyperparameter tuning: Limiting tree depth improved test performance for the tested Decision Tree configurations.
- Random Forest improvement: Random Forest outperformed a single Decision Tree for both regression and classification.
- Baseline performance: Linear Regression and Logistic Regression remained the strongest models for their respective tasks on the selected test split.
- Feature importance: Living area, lot size, and location were important predictors in both Random Forest models.
- Model complexity: More complex models do not automatically generalise better or achieve higher test scores.

## Tools

The project was implemented in Python using Google Colab.

Main libraries:

- `pandas` and `numpy` — data manipulation and numerical operations
- `matplotlib` and `seaborn` — visualisation
- `scikit-learn` — preprocessing, pipelines, tree-based models, and evaluation

The implementation uses `DecisionTreeRegressor`, `DecisionTreeClassifier`, `RandomForestRegressor` and `RandomForestClassifier`, together with evaluation metrics from `sklearn.metrics`.

## Limitations

The results are specific to the dataset, preprocessing procedure, and train/test split used in the experiments.

Only a limited number of hyperparameter configurations were tested, and evaluation was based on a single held-out test set. Further experiments using cross-validation and a broader hyperparameter search could provide more robust performance estimates.

Random Forest feature importance can also be influenced by the characteristics of the data and should not be interpreted as causal evidence.

## Possible improvements

Future work could investigate:

- Cross-validation for more reliable model evaluation
- Grid search or random search for more systematic hyperparameter tuning
- Gradient Boosting models for comparison with Random Forest
- Permutation importance to provide an alternative assessment of feature importance
- Further investigation of prediction errors and model generalisation


```


```
