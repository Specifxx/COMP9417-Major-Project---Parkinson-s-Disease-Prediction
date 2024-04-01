import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from collections import defaultdict

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


"""
This file trains all initial models using only visit_month
Outputs the train and validation scores for each model to stdout
"""

DATA_DIR = "supplied-files"
OUTPUT_DIR = "output"

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

train_clinical_data = pd.read_csv(DATA_DIR + "/train_clinical_data.csv")
train_peptides = pd.read_csv(DATA_DIR + "/train_peptides.csv")
train_proteins = pd.read_csv(DATA_DIR + "/train_proteins.csv")
supplemental_clinical_data = pd.read_csv(DATA_DIR + "/supplemental_clinical_data.csv")

def smape(y_true, y_pred):
    """
    - Calculates the score given truth and prediction values
    - Competition uses 'SMAPE+1' to adjust for low (<1.0) updrs scores
    """
    if len(y_true) != len(y_pred):
      raise ValueError(f"diff lengths for true ({len(y_true)}) and pred ({len(y_pred)})")
    y_true = 1+y_true
    y_pred = 1+y_pred
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))



"""
First prepare the training data, adding a 'target_month' column
so our models can be trained on number of months after first visit
to predict the UPDR scores
"""

def prepare_train_data(data):
    """
    Adds a target_month column to clinical data
    This is simply visit_month + month_offset
    """
        
    data = data.drop('upd23b_clinical_state_on_medication', axis=1)

    train = data.copy()

    patient_data = defaultdict(dict)

    for idx, row in data.iterrows():
      patient_id = row["patient_id"]
      visit_month = row["visit_month"]

      patient_data[patient_id][visit_month] = {
          "updrs_1": row["updrs_1"],
          "updrs_2": row["updrs_2"],
          "updrs_3": row["updrs_3"],
          "updrs_4": row["updrs_4"],
      }

    train = data.copy()
    train["target_month"] = train["visit_month"]

    for idx, row, in data.iterrows():
        visit_id = row["visit_id"]
        patient_id = row["patient_id"]
        visit_month = row["visit_month"]
        for month_offset in [6, 12, 24]:
          month = visit_month + month_offset
          if month in patient_data[patient_id]:
            # r = row.copy() does not preserve dtype
            r = {
                "visit_id": visit_id,
                "visit_month": visit_month,
                "target_month": visit_month + month_offset,
                "patient_id": patient_id,
                "updrs_1": patient_data[patient_id][month]["updrs_1"],
                "updrs_2": patient_data[patient_id][month]["updrs_2"],
                "updrs_3": patient_data[patient_id][month]["updrs_3"],
                "updrs_4": patient_data[patient_id][month]["updrs_4"],
            }

            train = pd.concat([train, pd.DataFrame([r])], ignore_index=True)

    return train

train = prepare_train_data(train_clinical_data)

print("\n============================== TRAIN DATA ==================================")
print(train.head())
print(train.tail())



"""
Now train models on this data, including:
    - LightGBM
    - XGBoost
    - LinearRegression
Using GroupKFolds Cross Validation -> grouped on patient_id
"""

pd.options.mode.chained_assignment = None # surpresss copy warning (we dont care abt modifying the df)

train_copy = train.copy()

groups = train_copy["patient_id"]

n_folds = 5

features = ["target_month"]

targets = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

skf = GroupKFold(n_folds)

X = train_copy[features]

train_scores = defaultdict(lambda: defaultdict(list))
val_scores = defaultdict(lambda: defaultdict(list))
scores = defaultdict(lambda: defaultdict(list))
models = defaultdict(lambda: defaultdict(list))

print("\n============================== TRAINING AND TESTING MODELS ==================================")

for target in targets:

    y = train_copy[target]
    
    print(f"TARGET: {target}")
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y, groups)):
        train_data = train_copy.iloc[train_index]
        validation_data = train_copy.iloc[test_index]
        
        # Drop rows where the current target is missing
        train_data.dropna(subset=[target], inplace=True)

        # Split data into train/validation sets
        X_train, X_val = train_data[features], validation_data[features]
        y_train, y_val = train_data[target], validation_data[target]

        lgbm_model = LGBMRegressor(random_state=23)
        xgb_model = XGBRegressor(
            random_state=23)
        linear_model = LinearRegression()
        
        lgbm_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        linear_model.fit(X_train, y_train)
        
        lgbm_pred = lgbm_model.predict(X_val)
        xgb_pred = xgb_model.predict(X_val)
        linear_pred = linear_model.predict(X_val)
        
        lgbm_score = smape(lgbm_pred, y_val).round(2)
        xgb_score = smape(xgb_pred, y_val).round(2)
        linear_score = smape(linear_pred, y_val).round(2)
        
        scores['lgbm'][target].append(lgbm_score)
        scores['linear'][target].append(linear_score)
        scores['xgb'][target].append(xgb_score)
        
        models['lgbm'][target].append(lgbm_model)
        models['linear'][target].append(linear_model)
        models['xgb'][target].append(xgb_model)
        
        print(f"\tFOLD {fold}")
        print(f"\t\tLGBM val score: {lgbm_score}")
        print(f"\t\tXGB val score: {xgb_score}")
        print(f"\t\tLinear val score: {linear_score}")
        

for target in targets:
    print(f"TARGET: {target}")
    print(f"\t LGBM score: {np.array(scores['lgbm'][target]).mean().round(2)}")
    print(f"\t XGB score: {np.array(scores['xgb'][target]).mean().round(2)}")
    print(f"\t Linear score: {np.array(scores['linear'][target]).mean().round(2)}")
    
print(f"OVERALL SCORES")
print(f"\t LGBM score: {np.array([np.array(scores['lgbm'][target]).mean() for target in targets]).mean().round(2)}")
print(f"\t XGB score: {np.array([np.array(scores['xgb'][target]).mean() for target in targets]).mean().round(2)}")
print(f"\t Linear score: {np.array([np.array(scores['linear'][target]).mean() for target in targets]).mean().round(2)}")
