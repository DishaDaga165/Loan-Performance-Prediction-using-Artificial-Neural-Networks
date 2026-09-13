# Loan Performance Prediction using Artificial Neural Networks

This project uses historical Prosper loan data to study whether borrower and loan characteristics can be used to predict loan performance using Artificial Neural Networks (ANNs).

The goal was to build the complete modelling workflow rather than only train a neural network — from data cleaning and exploratory analysis to leakage-safe preprocessing, model evaluation, optimizer comparison, and ANN sensitivity experiments.

---

## Dataset and Problem

The Prosper Loan Dataset contains historical information about borrowers, their credit profiles, loan characteristics, and loan performance.

After cleaning and retaining resolved loans, the final dataset contained over **40,000 loan records**.

I used 10 numerical variables as model inputs:

- Stated Monthly Income
- Debt-to-Income Ratio
- Credit Score
- Bankcard Utilization
- Available Bankcard Credit
- Revolving Credit Balance
- Amount Delinquent
- Delinquencies in the Last 7 Years
- Original Loan Amount
- Borrower Rate

The ANN predicts two continuous outputs:

**Realized Interest Ratio**

`LP_InterestandFees / LoanOriginalAmount`

This represents the interest and fees realized relative to the original loan amount.

**Net Principal Loss Ratio**

`LP_NetPrincipalLoss / LoanOriginalAmount`

This represents the net principal loss relative to the original amount lent.

Using ratios instead of absolute amounts makes outcomes across loans of different sizes easier to compare.

---

## Data Preparation and EDA

Before modelling, I explored the dataset to understand its quality and structure.

The analysis included:

- Missing value and duplicate checks
- Descriptive statistics
- Histograms for input and output variables
- Boxplots for identifying potential outliers
- Pearson correlation analysis
- Correlation heatmap
- Input-output correlation analysis
- Scatter plots between each input and both outputs

Potential outliers were visualized but not automatically removed, since extreme financial observations can represent genuine borrower behaviour.

### Correlation Analysis

<!-- ADD CORRELATION HEATMAP HERE -->

<img width="1200" height="1000" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/bba23476-2e5d-446e-9f2f-38d8e613e773" />


The dataset was then divided into:

- **70% Training**
- **15% Validation**
- **15% Testing**

Min-max normalization was applied using parameters calculated **only from the training data**. The same training minimum and maximum values were then used to transform validation and testing data.

This prevents information from unseen data from leaking into model preparation.

---

## ANN Model

The baseline model was a feed-forward neural network with the following architecture:

`10 Inputs → 64 ReLU → 32 ReLU → 2 Linear Outputs`

The two hidden layers use ReLU activation, while the output layer uses linear activation because both targets are continuous variables.

The model was trained using:

- Adam optimizer
- Learning rate = 0.001
- Mean Squared Error (MSE) loss
- Batch size = 32
- Early stopping based on validation loss

Model performance was evaluated using **MSE and R²** on both training and unseen testing data.

Metrics were also calculated separately for both outputs because the two loan outcomes have different levels of predictability.

---

## Adam vs RMSProp

I trained a second version of the same ANN using the **RMSProp optimizer**.

To make the comparison fair, the architecture, data split, normalization, learning rate, batch size, and loss function were kept unchanged. The optimizer was the main variable being changed.

The two models were compared using:

- Training MSE
- Training R²
- Testing MSE
- Testing R²
- Best validation MSE
- Best validation epoch
- Training and validation loss curves

### Optimizer Comparison

<!-- ADD ADAM VS RMSPROP VALIDATION LOSS GRAPH HERE -->

<img width="2400" height="1500" alt="adam_vs_rmsprop_validation_loss" src="https://github.com/user-attachments/assets/28c7803d-d513-4d0f-9113-6c40b310ed9e" />



Adam and RMSProp produced broadly similar results, suggesting that changing the optimizer alone did not substantially change the predictive limits of the current feature and target setup.

---

## ANN Sensitivity Analysis

I then studied how different ANN design choices affected model performance.

Instead of writing separate training code for every model, I created a reusable experiment function that builds, trains, evaluates, and generates predictions for a given ANN configuration.

Five parameters were studied with three variations each:

| Parameter | Variations |
|---|---|
| Hidden Layers | 1, 2, 3 |
| Hidden Nodes | (32,16), (64,32), (128,64) |
| Activation Function | ReLU, tanh, sigmoid |
| Training Epochs | 25, 50, 100 |
| Training Sample Size | 25%, 50%, 100% |

This resulted in **15 controlled ANN experiments**.

Only one parameter was changed at a time while the remaining settings were kept constant. This made it easier to understand which modelling choice was responsible for a change in performance.

For every configuration, I compared predicted and true values for both outputs on both training and testing data.

### Predicted vs True Outputs

<!-- ADD THE BEST/REPRESENTATIVE PREDICTED VS TRUE 2×2 FIGURE HERE -->

<img width="2370" height="1092" alt="image" src="https://github.com/user-attachments/assets/662df094-d6d6-4d90-a3cb-d090f3837ef3" />


The diagonal reference line represents perfect prediction. Predictions closer to this line indicate better agreement between the ANN and the actual loan outcome.

---

## What I Learned

This project showed me that building the neural network itself is only one part of an ML problem.

How the targets are defined, how the data is split, preventing leakage, selecting evaluation metrics, and checking performance on unseen data can matter just as much as changing the ANN architecture.

The experiments also showed that simply increasing network complexity does not automatically produce a better model.

The current ANN captures useful signal from borrower and loan characteristics, but it explains only part of the variation in the two loan outcomes, particularly principal loss.

That leads to a more interesting next question than simply making the ANN deeper:

**Can a different modelling approach or better problem formulation predict these outcomes more effectively?**

---

## Next Steps

The next version of this project will extend the ANN analysis into a broader lending decision-support system by:

- Benchmarking the ANN against other models for tabular financial data
- Identifying the strongest predictive model rather than assuming the ANN is best
- Adding model explainability to understand the drivers behind individual predictions
- Translating predicted interest and loss into a simple risk/value framework
- Building an interactive interface for exploring borrower-level predictions

The aim is to move from simply predicting loan outcomes to understanding how those predictions could support an actual lending decision.
