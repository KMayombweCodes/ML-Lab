# Week 3: Model evaluation

We now extend the previous weeks' house price ML work by focusing on **model evaluation, baseline comparisons, data leakage prevention, and cross-validation**. Using small numerical examples and the housing dataset, we demonstrate how to assess classification and regression performance, compare models fairly, and interpret prediction errors.

The implementation is provided in the accompanying *notebook*, using Python and scikit-learn.

---

## Project overview

The project covers the following topics:

- **Baseline models** — comparing trained models against simple prediction strategies
- **Data leakage** — identifying and excluding features that reveal the target
- **Train/test splitting** — reserving unseen observations for final evaluation
- **5-fold cross-validation** — evaluating model performance across multiple training and validation splits
- **Model comparison** — comparing a majority-class baseline, Logistic Regression, and Random Forest
- **Confusion matrices** — interpreting correct predictions, false positives, and false negatives
- **Performance metrics** — evaluating models using accuracy, precision, recall, F1 Score, and MAE

---

## Key results

Logistic Regression achieved the best mean F1 Score during 5-fold cross-validation and was selected for final testing.

| Evaluation | Model | Result |
|---|---|---:|
| Cross-validation mean accuracy | Logistic Regression | 92.2% |
| Cross-validation mean F1 Score | Logistic Regression | 93.8% |
| Final test accuracy | Logistic Regression | 93.3% |
| Final test precision | Logistic Regression | 94.7% |
| Final test recall | Logistic Regression | 94.7% |
| Final test F1 Score | Logistic Regression | 94.7% |

The final test confusion matrix contained **80 true negatives, 144 true positives, 8 false positives, and 8 false negatives**.

The notebook also demonstrates that a regression model can outperform a mean-price baseline, achieving a mean absolute error of €15,000 compared with €50,000 for the baseline in a small illustrative example.

---

## Key findings

- Baseline models provide a reference for determining whether a trained model offers meaningful improvement.
- Five-fold cross-validation provides a more informative assessment of performance than relying on a single split.
- Preprocessing must be fitted only on the relevant training data to prevent leakage.
- Confusion matrices reveal the types of errors that accuracy alone cannot explain.
- The final test set should remain separate from model selection to provide an independent assessment of generalisation.

---

