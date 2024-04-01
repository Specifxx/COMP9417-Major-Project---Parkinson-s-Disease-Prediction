import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from collections import defaultdict

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


"""
This file trains an LGBM model using visit_month and month_offset
Uses GridSearch to find the optimal parameters for each target
    - If you want to run faster, use RandomizedSearch
These parameters are used for the models on each target
Outputs the validation scores to stdout
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
First prepare the training data, adding a 'month_offset' column
so our models can be trained on visit_month and the number of months after
to predict the UPDR scores
"""
def prepare_train_data(data):
        
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
    train["month_offset"] = 0

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
                "month_offset": month_offset,
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
Now conduct a Grid/Randomised search to find the best parameters
Warning: Grid takes a while
"""

train_copy = train.copy()

groups = train_copy["patient_id"]

n_folds = 5

features = ["visit_month", "month_offset"]

targets = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

X = train_copy[features]

scores = defaultdict(list)

model_params = {}

for target in targets:
    y = train_copy[target]
    
    print(f"TARGET: {target}")
    
    lgbm_model = LGBMRegressor()
    
    param_lgb = [{
        'n_estimators': [20, 50, 75, 100, 200, 400],
        'learning_rate': [0.002, 0.01, 0.05, 0.1, 0.5],
        'max_depth': [-1, 3, 5, 7, 9, 13, 17],
        'num_leaves': [20, 31, 40, 50, 60, 70]
    }]
    
    cv = GroupKFold(n_splits=5)
    
    lgb = RandomizedSearchCV(lgbm_model, param_lgb, cv = cv, scoring=make_scorer(smape), verbose = -1)
    # lgb = GridSearchCV(lgbm_model, param_lgb, cv = cv, scoring=make_scorer(smape), verbose = -1)
    
    lgb.fit(X, y, groups=groups)
    
    print('light gbm best param',lgb.best_params_)
    print('light gbm best score',lgb.best_score_)

    model_params[target] = lgb.best_params_


from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

models = defaultdict(list)

train_copy = train.copy()

groups = train_copy["patient_id"]
n_folds = 5
features = ["visit_month", "month_offset"]
# features = ["target_month"]
targets = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

skf = GroupKFold(n_folds)

X = train_copy[features]

scores = defaultdict(list)
train_scores = defaultdict(list)

for target in targets:
    y = train_copy[target]
    
    print(f"TARGET: {target}")
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        
        model = LGBMRegressor(**model_params[target])
        
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        score = smape(pred, y_val)

        train_pred = model.predict(X_train)
        train_score = smape(train_pred, y_train)
        
        train_scores[target].append(train_score)
        
#         if target == 'updrs_4':
#             scores[target].append(smape(np.zeros(y_train.shape), y_train))
#             continue
            
        scores[target].append(score)
        models[target].append(model)
        
        print(f"\tFOLD {fold}: TRAIN SCORE {train_score.round(2)}")
        print(f"\tFOLD {fold}: VALIDATION SCORE {score.round(2)}\n")
        
        
for target in targets:
    print(f"TARGET: {target}, TRAIN SCORE: {np.array(train_scores[target]).mean().round(2)}")
    print(f"TARGET: {target}, VALIDATION SCORE: {np.array(scores[target]).mean().round(2)}\n")

print(f"TOTAL OOF SMAPE: {np.array([score for target in targets for score in scores[target]]).mean().round(2)}")
