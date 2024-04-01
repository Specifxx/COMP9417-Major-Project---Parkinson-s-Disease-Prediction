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
This file explores handpicking the proteins with the highest correlation
to specific UDPR scores.

There are two available techniques to use when the protein readings are not
available in the test data:
    - A technique using the rolling max median for each UPDR score vs month
    - An LGBM model trained on visit_month and month_offset

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
Now prepare the training data including the hand-picked proteins,
training only on data where those protein readings are available

See the top proteins in the EDA output
"""
updrs_1_protein = 'P04180'
updrs_2_protein = 'P04180'
updrs_3_protein = 'O00533'
feature_proteins = ['P04180', 'O00533']


protein_dict = defaultdict(dict)

for idx, (visit_id, visit_month, patient_id, UniProt, NPX) in train_proteins.iterrows():
    protein_dict[UniProt][visit_id] = NPX

def prepare_prot_train_data(clinical_data, protein_dict, target_prot, target):
    """
    Adds the NPX readings of target_prot into clinical data
    Only keeps visits for which that protein reading was available
    """
        
    clinical_data = clinical_data.drop('upd23b_clinical_state_on_medication', axis=1)

    train = clinical_data.copy()
    
    patient_data = defaultdict(dict)
    
    # get all visits with the target protein
    visits = set([v_id for v_id in protein_dict[target_prot]])
    train = train.loc[train["visit_id"].isin(visits)]

    for idx, row in train.iterrows():
      patient_id = row["patient_id"]
      visit_month = row["visit_month"]

      patient_data[patient_id][visit_month] = {
          "visit_id": row["visit_id"],
          "updrs_1": row["updrs_1"],
          "updrs_2": row["updrs_2"],
          "updrs_3": row["updrs_3"],
          "updrs_4": row["updrs_4"],
      }

    train = train.copy()
    train["month_offset"] = 0
    train["NPX"] = train.visit_id.apply(lambda x: protein_dict[target_prot][x])
    
    # add month offset to data
    for idx, row, in train.iterrows():
        visit_id = row["visit_id"]
        patient_id = row["patient_id"]
        visit_month = row["visit_month"]
        for month_offset in [6, 12, 24]:
          target_month = visit_month + month_offset
          if target_month in patient_data[patient_id] and patient_data[patient_id][target_month]['visit_id'] in visits:
            # r = row.copy() does not preserve dtype
            r = {
                "visit_id": visit_id,
                "visit_month": visit_month,
                "month_offset": month_offset,
                "patient_id": patient_id,
                "updrs_1": patient_data[patient_id][target_month]["updrs_1"],
                "updrs_2": patient_data[patient_id][target_month]["updrs_2"],
                "updrs_3": patient_data[patient_id][target_month]["updrs_3"],
                "updrs_4": patient_data[patient_id][target_month]["updrs_4"],
                "NPX": protein_dict[target_prot][visit_id]
            }

            train = pd.concat([train, pd.DataFrame([r])], ignore_index=True)

    return train

prot_train_updrs_1 = prepare_prot_train_data(train_clinical_data, protein_dict, target_prot=updrs_1_protein, target='updrs_1')
prot_train_updrs_2 = prepare_prot_train_data(train_clinical_data, protein_dict, target_prot=updrs_2_protein, target='updrs_2')
prot_train_updrs_3 = prepare_prot_train_data(train_clinical_data, protein_dict, target_prot=updrs_3_protein, target='updrs_3')
print("\n============================== TRAIN PROTEIN DATA (uprs2) ==================================")
print(prot_train_updrs_2.head())
print(prot_train_updrs_2.tail())

"""
Train models on the NPX value + visit month and month offset
"""
prot_features = ['visit_month', 'month_offset', 'NPX']

# TRAIN UPDRS 1
prot_model_updrs_1 = LGBMRegressor(random_state=23)
target = 'updrs_1'

X = prot_train_updrs_1[prot_features]
y = prot_train_updrs_1[target]
prot_model_updrs_1.fit(X, y)

# TRAIN UPDRS 2
prot_model_updrs_2 = LGBMRegressor(random_state=23)
target = 'updrs_2'

X = prot_train_updrs_2[prot_features]
y = prot_train_updrs_2[target]
prot_model_updrs_2.fit(X, y)

# TRAIN UPDRS 3
prot_model_updrs_3 = LGBMRegressor(random_state=23)
target = 'updrs_3'

X = prot_train_updrs_3[prot_features]
y = prot_train_updrs_3[target]
prot_model_updrs_3.fit(X, y)

"""
Functions to return predictions based on above models
If the visit_id didn't include the feature protein, then predict NaN
"""
def get_prot_updrs_1_predictions(X):
    preds = []
    for idx, row in X.iterrows():
        if pd.notna(row[updrs_1_protein]):
            preds.append(prot_model_updrs_1.predict([[row['visit_month'], row['month_offset'], row[updrs_1_protein]]])[0])
        else:
            preds.append(np.nan)
    return preds

def get_prot_updrs_2_predictions(X):
    preds = []
    for idx, row in X.iterrows():
        if pd.notna(row[updrs_2_protein]):
            preds.append(prot_model_updrs_2.predict([[row['visit_month'], row['month_offset'], row[updrs_2_protein]]])[0])
        else:
            preds.append(np.nan)
    return preds

def get_prot_updrs_3_predictions(X):
    preds = []
    for idx, row in X.iterrows():
        if pd.notna(row[updrs_3_protein]):
            preds.append(prot_model_updrs_3.predict([[row['visit_month'], row['month_offset'], row[updrs_3_protein]]])[0])
        else:
            preds.append(np.nan)
    return preds

"""
Prepare dataframe of UPDRs medians based on months since first visit
Applies a rolling max for the specified targets

Inspired by https://www.kaggle.com/code/gunesevitan/amp-pdpp-baseline
"""
target_columns_clinical = ['updrs_1'] # targets not trained on supp data
target_columns_clinical_and_supp = ['updrs_2', 'updrs_3', 'updrs_4']

target_visit_month_medians_clinical = train_clinical_data.groupby('visit_month')[target_columns_clinical].median()

# merge supp clinical data
target_visit_month_medians_clinical_and_supp = pd.concat((
    train_clinical_data,
    supplemental_clinical_data
), axis=0).groupby('visit_month')[target_columns_clinical_and_supp].median()

# remove outlier months
outlier_months = [5]
for m in outlier_months:
    target_visit_month_medians_clinical_and_supp.drop(m, inplace=True)

# Concatenate visit_month medians of targets
target_visit_month_medians = pd.concat((
    target_visit_month_medians_clinical,
    target_visit_month_medians_clinical_and_supp
), axis=1, ignore_index=False)

target_columns_default = ['updrs_1'] # move targets here if you don't want rolling max
target_columns_expanding = ['updrs_2', 'updrs_3', 'updrs_4']

target_visit_month_expanding = target_visit_month_medians[target_columns_expanding].expanding(min_periods=1).max()

target_visit_month_medians = pd.concat((
    target_visit_month_expanding,
    target_visit_month_medians[target_columns_default]
), axis=1, ignore_index=False)

def get_median_prediction(X, target):
    target_months = X["visit_month"] + X["month_offset"]
    preds = []
    for idx, month in target_months.items():
        if month in target_visit_month_medians.index:
            preds.append(target_visit_month_medians.loc[month, target])
        else:
            # Find the closest visit_month value from the baselines table
            target_visit_month_median_idx = np.argmin(np.abs(target_visit_month_medians.index - month))
            # Write the closest visit_month value to the unseen visit_month
            preds.append(target_visit_month_medians.loc[target_visit_month_median_idx, target])
    return np.array(preds)

print("\n============================== MEDIANS DATAFRAME ==================================")
print(target_visit_month_medians)

models = defaultdict(list)

train_copy = train.copy()

groups = train_copy["patient_id"]
n_folds = 5

features = ["visit_month", "month_offset"]
prot_features = ["visit_month", "month_offset", "NPX"]

targets = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

skf = GroupKFold(n_folds)


# add protein stuff to train_copy
for protein in protein_dict:
    train_copy[protein] = train_copy["visit_id"].apply(
           lambda visit_id: protein_dict[protein][visit_id] if visit_id in protein_dict[protein] else np.nan
    )
    
X = train_copy[features + feature_proteins]

scores = defaultdict(list)
train_scores = defaultdict(list)

print("\n============================== TRAINING AND TESTING MODELS ==================================")

"""
Able to use either the medians method OR the LGBM regressor
where protein data is unavailable
"""

for target in targets:
    y = train_copy[target]
    
    print(f"TARGET: {target}")
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        
        model = LGBMRegressor(random_state=23)
        
        model.fit(X_train[features], y_train)
        
        # pred = np.array(model.predict(X_val[features])) #! USE FOR LGBM
        pred = get_median_prediction(X_val[features], target) #! USE FOR MEDIANS
        
        if target == 'updrs_2':
            prot_preds = np.array(get_prot_updrs_2_predictions(X_val[features + [updrs_2_protein]]))
            pred = np.where(~np.isnan(prot_preds),prot_preds,pred)
        elif target == 'updrs_3':
            prot_preds = np.array(get_prot_updrs_3_predictions(X_val[features + [updrs_3_protein]]))
            pred = np.where(~np.isnan(prot_preds),prot_preds,pred)
        elif target == 'updrs_1':
            prot_preds = np.array(get_prot_updrs_1_predictions(X_val[features + [updrs_1_protein]]))
            pred = np.where(~np.isnan(prot_preds),prot_preds,pred)

        score = smape(pred, y_val)
        
        train_pred = model.predict(X_train[features])
        train_score = smape(train_pred, y_train)
        
        scores[target].append(score)
        train_scores[target].append(train_score)
        
        models[target].append(model)
        
        print(f"\tFOLD {fold}: TRAIN SCORE {train_score.round(2)}")
        print(f"\tFOLD {fold}: VALIDATION SCORE {score.round(2)}\n")
        
        
for target in targets:
    print(f"TARGET: {target}, TRAIN SCORE: {np.array(train_scores[target]).mean().round(2)}")
    print(f"TARGET: {target}, VALIDATION SCORE: {np.array(scores[target]).mean().round(2)}\n")
print(f"TOTAL OOF SCORE: {np.array([score for target in targets for score in scores[target]]).mean().round(2)}")
