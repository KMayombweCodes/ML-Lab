# *House price* prediction and classification

This project applies machine learning to a housing dataset containing information about 1,208 properties and 19 variables. The dataset includes realistic data-quality issues such as missing values, inconsistent categorical entries, malformed numerical values, and duplicate property records.

The project addresses two machine learning problems:

1. **House price prediction** using Linear Regression
2. **High-price classification** using Logistic Regression

The complete implementation is provided in the accompanying Jupyter/Google Colab notebook, while the detailed methodology and discussion are presented in the project report.

---

## Project overview

The project follows a standard machine learning workflow:

**Data Understanding → Data Cleaning → Exploratory Data Analysis → Preprocessing → Model Training → Evaluation → Interpretation**

The raw dataset was cleaned before modelling. Missing and invalid values were handled, categorical values were standardised, malformed numerical values were converted, and duplicate property records were removed.

After cleaning, 1,200 observations remained for modelling.

---

## Dataset

The dataset contains information about properties, including:

- Location and neighborhood
- Property type and condition
- Bedrooms and bathrooms
- Living area and lot size
- Year built
- Parking spaces
- Distance from the city centre
- School and crime ratings
- Garden and balcony availability
- Energy rating
- Sale price

The main target variables are:

- `sale_price_eur` — continuous house sale price
- `high_price` — binary classification target

A property is considered high-priced when:

    high_price = 1  if sale_price_eur >= €500,000
    high_price = 0  if sale_price_eur <  €500,000

---

## EDA

Exploratory data analysis was performed to understand the distribution of house prices and investigate relationships between the available features.

The strongest numerical relationships with sale price were:

| Feature | Correlation |
|---|---:|
| Living area | 0.730 |
| Bedrooms | 0.618 |
| Lot size | 0.604 |
| Bathrooms | 0.422 |
| Parking spaces | 0.415 |
| Has garden | 0.408 |

**Living area** had the strongest numerical correlation with sale price.

There were also substantial differences between locations and property types. For example, the approximate median house price ranged from €370,000 in Leipzig to €1,040,000 in Munich.

These results suggest that both numerical characteristics and categorical information such as location and property type are useful for predicting house prices.

---

# Part A — House price prediction

The first task is a regression problem. The objective is to predict the continuous variable:

    sale_price_eur

A **Linear Regression** model was trained using the available property characteristics.

`house_id` was excluded because it is an identifier rather than a meaningful predictor. Variables derived from the sale price were also excluded to prevent *target leakage**.

The dataset was split into 80% training data and 20% test data, using `random_state=42` for reproducibility.

### Results

| Metric | Result |
|---|---:|
| **MAE** | €100,874 |
| **RMSE** | €140,585 |
| **R²** | 0.8796 |

The model achieved an **R² of 0.8796**, meaning that approximately **88% of the variation in house prices** in the test set is explained by the model.

The MAE indicates an average absolute prediction error of approximately **€101,000**.

---

# Part B — High-price classification

The second task is a **binary classification problem**.

A house is classified as high-priced when its sale price is at least **€500,000**. A **Logistic Regression** model was used to predict this classification from the available property characteristics.

To avoid **data leakage**, `sale_price_eur` was not used as an input feature because `high_price` is directly derived from it.

Numerical variables were standardised and categorical variables were one-hot encoded before training the model.

### Results

| Metric | Result |
|---|---:|
| **Accuracy** | 92.92% |
| **Precision** | 94.12% |
| **Recall** | 94.74% |
| **F1 Score** | 94.43% |

The model correctly classified approximately **93% of the test observations**.

Its recall of **94.74%** indicates that it successfully identified most of the houses that genuinely belonged to the high-price category.

### Confusion Matrix

    [[79,   9],
     [ 8, 144]]

Using rows as the actual classes and columns as the predicted classes:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| **Actual 0** | 79 | 9 |
| **Actual 1** | 8 | 144 |

---

## Data preprocessing

The dataset required several cleaning steps before modelling.

Malformed numerical values such as:

    "1 baths"
    "90m2"

were converted into usable numerical values.

Missing and invalid values were handled through the preprocessing pipeline. Inconsistent categorical values, including differences in spelling and capitalisation, were standardised.

Duplicate property records were identified while ignoring `house_id`. This resulted in **8 duplicate records being removed**.

The preprocessing included:

- Missing-value imputation
- Numerical value cleaning
- Categorical value standardisation
- One-hot encoding
- Numerical scaling for Logistic Regression
- Duplicate removal
- Train/test splitting

The preprocessing and models were implemented using **scikit-learn pipelines** to ensure consistent transformations between training and test data.

---

## Key findings

The analysis indicates that **living area, bedrooms, lot size, bathrooms, parking spaces, and garden availability** are among the more important numerical characteristics associated with house prices.

Location also plays an important role, with considerable differences in median prices between cities.

Overall, both models produced strong results for their respective tasks:

| Model | Task | Main Result |
|---|---|---:|
| Linear Regression | Price prediction | **R² = 0.8796** |
| Logistic Regression | High-price classification | **F1 = 0.9443** |

These scores should not be directly compared because the models solve different machine learning problems.

---

## Technologies

The project was implemented in **Python** using **Google Colab**.

Main libraries:

- `pandas` — data manipulation
- `numpy` — numerical operations
- `matplotlib` — visualisation
- `scikit-learn` — preprocessing, modelling, pipelines, and evaluation

---
The Python notebook contains the complete implementation, including data cleaning, exploratory analysis, preprocessing, model training and evaluation.

The PDF report provides the detailed methodology, mathematical background, interpretation, limitations, and discussion of possible improvements.

---

## Possible improvements

The current models provide a strong baseline. Future work could investigate:

- **Feature engineering** to create more informative predictors
- **K-fold cross-validation** for more robust model evaluation
- **Random Forest** and **Gradient Boosting** for non-linear relationships
- Further **feature selection**
- More detailed investigation of unusual or high-error observations

---


This project was developed as part of a machine learning assignment focused on practical data preprocessing, regression, classification, and model evaluation.
