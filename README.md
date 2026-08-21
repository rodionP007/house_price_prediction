# House Price Prediction

This project is a machine learning regression project based on the Kaggle **House Prices - Advanced Regression Techniques** dataset.

The goal of the project is to predict house sale prices using numerical and categorical features such as overall quality, living area, garage size, basement area, year built, neighborhood, and other property characteristics.

## Project Goals

The main goals of this project are:

* build a complete baseline machine learning pipeline for a regression task;
* perform exploratory data analysis;
* handle missing values;
* process numerical and categorical features;
* compare several regression models;
* tune model hyperparameters;
* select the best model based on RMSE;
* generate a Kaggle-compatible `submission.csv`;
* save the final trained model.

## Dataset

The project uses the Kaggle dataset:

**House Prices - Advanced Regression Techniques**

The dataset contains information about residential homes and their final sale prices.

Target variable:

```text
SalePrice
```

The original data files are not included in this repository. To reproduce the project, download the dataset from Kaggle and place the files into:

```text
data/raw/
```

Expected files:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/data_description.txt
data/raw/sample_submission.csv
```

## Project Structure

```text
house_price_prediction/
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── 01_eda_and_baseline.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Workflow

The project follows a standard machine learning workflow:

1. Load and inspect the data.
2. Analyze the target variable `SalePrice`.
3. Check missing values.
4. Split features into numerical and categorical groups.
5. Perform exploratory data analysis.
6. Create preprocessing pipelines.
7. Split the data into train and test sets.
8. Train baseline and machine learning models.
9. Compare models using regression metrics.
10. Tune selected models with cross-validation.
11. Select the final model.
12. Generate predictions for Kaggle `test.csv`.
13. Save the trained model.

## Preprocessing

The preprocessing pipeline handles numerical and categorical features separately.

Numerical features:

```text
SimpleImputer(strategy="median")
StandardScaler()
```

Categorical features:

```text
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
```

For tree-based models, scaling is not required, so a separate preprocessing pipeline without `StandardScaler` is used.

The technical column `Id` is excluded from model training because it is only an identifier and does not describe the house itself.

## Models

The following models were trained and compared:

* DummyRegressor
* LinearRegression
* Ridge
* Lasso
* Ridge with GridSearchCV
* Lasso with GridSearchCV
* RandomForestRegressor
* RandomForestRegressor with RandomizedSearchCV
* HistGradientBoostingRegressor

## Evaluation Metrics

The models were evaluated using:

* MAE
* MSE
* RMSE
* R2 Score

The main comparison metric is:

```text
RMSE
```

Lower RMSE means better prediction quality.

## Results

The baseline model produced the weakest result because it predicts the average house price for every object and does not use any features.

All machine learning models significantly outperformed the baseline, which shows that the dataset features contain useful information for predicting `SalePrice`.

Approximate model comparison:

| Model                               |    RMSE |     R2 |
| ----------------------------------- | ------: | -----: |
| Lasso                               | ~28,361 | ~0.895 |
| Best Lasso GridSearch               | ~28,388 | ~0.895 |
| Random Forest                       | ~28,555 | ~0.894 |
| Best Random Forest RandomizedSearch | ~28,983 | ~0.891 |
| Linear Regression                   | ~29,476 | ~0.887 |
| Ridge                               | ~29,844 | ~0.884 |
| HistGradientBoosting                | ~29,845 | ~0.884 |
| Best Ridge GridSearch               | ~30,655 | ~0.877 |
| Baseline                            | ~87,619 | ~0.000 |

Although the manually configured Lasso model showed a slightly lower RMSE on the test split, the difference compared to `Best Lasso GridSearch` is very small. The GridSearchCV version is methodologically more reliable because its hyperparameter was selected using cross-validation on the training data.

Final selected model:

```text
Best Lasso GridSearch
```

## How to Run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Place Kaggle data files into:

```text
data/raw/
```

Then open and run the notebook:

```text
notebooks/01_eda_and_baseline.ipynb
```

## Outputs

The notebook generates:

```text
data/processed/submission.csv
```

and saves the trained model to:

```text
models/final_model.joblib
```

These generated files are ignored by Git and can be recreated by running the notebook.

## Key Takeaways

This project demonstrates a complete beginner-friendly machine learning workflow for a regression problem:

* data loading;
* exploratory data analysis;
* missing value handling;
* preprocessing with `Pipeline` and `ColumnTransformer`;
* baseline comparison;
* model training;
* hyperparameter tuning;
* final model selection;
* prediction generation;
* model saving.

The project can be improved further by adding advanced feature engineering, logarithmic target transformation, CatBoost or LightGBM, model interpretation with feature importance or SHAP, and a small Streamlit or FastAPI application.

## Kaggle Score

The final selected model was submitted to the Kaggle competition **House Prices - Advanced Regression Techniques**.

Kaggle public score:

```text
0.13840
```

## Project Status

Version 1.0 is completed.

Planned improvements:
- advanced feature engineering;
- logarithmic target transformation;
- CatBoost or LightGBM;
- feature importance and SHAP;
- moving training code from notebook to `src/`;
- Streamlit or FastAPI demo;
- Docker support.
