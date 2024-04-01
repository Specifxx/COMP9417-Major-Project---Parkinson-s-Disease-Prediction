import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA

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

DATA_DIR = "supplied-files"
PCA_DIR = "PCA-dataset"

train_clinical = pd.read_csv(DATA_DIR + "/train_clinical_data.csv")
train_peptides = pd.read_csv(DATA_DIR + "/train_peptides.csv")
train_proteins = pd.read_csv(DATA_DIR + "/train_proteins.csv")
supplemental_clinical_data = pd.read_csv(DATA_DIR + "/supplemental_clinical_data.csv")

no_uni_peptieds = train_peptides['Peptide'].nunique()

#* MERGE DATASETS TOGETHER
protein_peptide_merged = pd.merge(train_proteins, train_peptides, on=['visit_id', 'UniProt'], how='left')
protein_peptide_merged = protein_peptide_merged.drop(['visit_month_y', 'patient_id_y'], axis=1)
protein_peptide_merged = protein_peptide_merged.rename(columns={'visit_month_x': 'visit_month', 'patient_id_x':'patient_id'})

# joining peptides and proteins to find no of unique combinations
protein_peptide_merged['pep_prot'] = protein_peptide_merged['Peptide'] + '_' + protein_peptide_merged['UniProt']
no_uni_pep_prot = protein_peptide_merged['pep_prot'].nunique()

unique_peptides_in_train = train_peptides['Peptide'].unique()


# creating features for train by using all ptotien and peptides values
protein_peptide_Abe = protein_peptide_merged.copy()
protein_peptide_Abe['pep_prot_Abe'] = protein_peptide_merged['Peptide'] + '_' + protein_peptide_merged['UniProt'] + '_Abe'

protein_peptide_NPX = protein_peptide_merged.copy()
protein_peptide_NPX['pep_prot_NPX'] = protein_peptide_merged['Peptide'] + '_' + protein_peptide_merged['UniProt'] + '_NPX'

protein_peptide_AbexNPX = protein_peptide_merged.copy()
protein_peptide_AbexNPX['pep_prot_AbexNPX'] = protein_peptide_merged['Peptide'] + '_' + protein_peptide_merged['UniProt'] + '_AbexNPX'
protein_peptide_AbexNPX['AbexNPX'] = protein_peptide_AbexNPX['PeptideAbundance'] * protein_peptide_AbexNPX['NPX']


PepAbundance = protein_peptide_Abe.pivot_table(index=['visit_id', 'visit_month'], columns='pep_prot_Abe', values='PeptideAbundance')
PepAbundance = PepAbundance.sort_values(by='visit_id')


PepNPX = protein_peptide_NPX.pivot_table(index=['visit_id', 'visit_month'], columns='pep_prot_NPX', values='NPX')
PepNPX = PepNPX.sort_values(by='visit_id')

PepAbundancexNPX = protein_peptide_AbexNPX.pivot_table(index=['visit_id', 'visit_month'], columns='pep_prot_AbexNPX', values='AbexNPX')
PepAbundancexNPX = PepAbundancexNPX.sort_values(by='visit_id')


merged1 = pd.merge(PepAbundance, PepNPX, on=['visit_id', 'visit_month'], how='left')
all_merged = pd.merge(merged1, PepAbundancexNPX, on=['visit_id', 'visit_month'], how='left')

# fill all absent peptides and protein with 0
all_merged = all_merged.fillna(0)
total_data = pd.merge(train_clinical, all_merged, on='visit_id', how='right')

#checking if meds have any effect on each updrs
on_meds = train_clinical[train_clinical['upd23b_clinical_state_on_medication'] == 'On']
off_meds = train_clinical[train_clinical['upd23b_clinical_state_on_medication'] == 'Off']
na_meds = train_clinical[train_clinical['upd23b_clinical_state_on_medication'].isna()]

# creating the target dataset for training progression in disease
targets = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']
expanded_targets = ['updrs+6', 'updrs+12', 'updrs+18', 'updrs+24', 'updrs+30']
y_dfs = []
for target in targets:
    target_dict = defaultdict(list)
    
    clinical_df = train_clinical[['visit_id', 'patient_id', 'visit_month', target]]
    clinical_dict = clinical_df.to_dict(orient='dict')
    for key, visit in clinical_dict['visit_id'].items():
        patient = clinical_dict['patient_id'][key]
        initial_visit = clinical_dict['visit_month'][key]
        updrs_0 = clinical_dict[target][key]
        target_dict['visit_id'].append(visit)
        target_dict['patient_id'].append(patient)
        target_dict['visit_month'].append(initial_visit)
        target_dict['updrs+0'].append(updrs_0)
        for key1, visit1 in clinical_dict['visit_id'].items():
            for time in [6, 12, 18, 24, 30]:
                if clinical_dict['patient_id'][key1] == patient and clinical_dict['visit_month'][key1] == initial_visit+time:
                    target_dict[f'updrs+{time}'].append(clinical_dict[target][key1])
        
        for time in expanded_targets:
            if len(target_dict[time]) < key+1:
                target_dict[time].append(None)
    
    target_df = pd.DataFrame(target_dict)
    y_dfs.append(target_df)
    target_df.to_csv(PCA_DIR + f"/target_{target}.csv", index=True)
 
 
# for storing all the models   
models = {}
ss = StandardScaler()

for index, target in enumerate(targets):
    
    X_y_1 = pd.merge(all_merged, y_dfs[index], on='visit_id', how='left')
    X_y_1_copy = X_y_1.copy()
    
    #drop all 0 month when NA value
    months_progression = ['updrs+0', 'updrs+6', 'updrs+12', 'updrs+18', 'updrs+24', 'updrs+30']
    for month_offset in months_progression:
        X_y_1_0 = X_y_1_copy.dropna(subset=[month_offset])
        X_train = X_y_1_0.drop(columns=['visit_id','patient_id', 'updrs+0', 'updrs+6', 'updrs+12', 'updrs+18', 'updrs+24', 'updrs+30'], axis = 1)

        groups = X_y_1_0["patient_id"]
        Y_train_0 = X_y_1_0[month_offset]

        X_train = X_train.sort_index(axis=1)
        
        # Normalize the data
        X_train_trnsfrmd = ss.fit_transform(X_train)
        y = Y_train_0
      
      
        # Use PCA to get the top 300 features
        pca = PCA(n_components=300) # choose the number of components to keep
        pca.fit(X_train_trnsfrmd)

        # # Transform the data into the reduced dimensionality space
        X_train_pca = pca.transform(X_train_trnsfrmd)
  
        print(f"TARGET: {target}{month_offset}")
        
        gkf = GroupKFold(n_splits=5)
        
        # Using a Random Forest Regressor model
        model = RandomForestRegressor(
                n_estimators=100,
                min_samples_split=36,
                min_samples_leaf=15,
                max_depth=None,
                bootstrap=True
            )
        
        # Train and evaluate model for each fold
        scores = cross_val_score(model, X_train_pca, y, groups=groups, cv=gkf, scoring=make_scorer(smape))
        print(scores)
    
        model.fit(X_train_pca, y)
        models[f"{target}_{month_offset}"] = model
        
        
             
# funtion to get features dataframe from test dataset 
"""
Add features to the test dataset
Includes Protiein/Peptide NPX/abundance and NPX/Abundance ratio
"""
def get_features(train, train_proteins, train_peptides, unique_peptides):
    # only allow peptides seen in the train dataset
    train_ids = train[['visit_id', 'visit_month', 'patient_id']]
    train_ids = train_ids.drop_duplicates()
    
    train_peptides = train_peptides[train_peptides['Peptide'].isin(unique_peptides)]

    protein_peptide_merged = pd.merge(train_proteins, train_peptides, on=['visit_id', 'UniProt'], how='left')
    protein_peptide_merged = protein_peptide_merged.drop(['visit_month_y', 'patient_id_y'], axis=1)
    protein_peptide_merged = protein_peptide_merged.rename(columns={'visit_month_x': 'visit_month', 'patient_id_x':'patient_id'})

    
    protein_peptide_Abe = protein_peptide_merged.copy()
    protein_peptide_Abe['pep_prot_Abe'] = protein_peptide_merged['Peptide'] + '_' + protein_peptide_merged['UniProt'] + '_Abe'

    protein_peptide_NPX = protein_peptide_merged.copy()
    protein_peptide_NPX['pep_prot_NPX'] = protein_peptide_merged['Peptide'] + '_' + protein_peptide_merged['UniProt'] + '_NPX'

    protein_peptide_AbexNPX = protein_peptide_merged.copy()
    protein_peptide_AbexNPX['pep_prot_AbexNPX'] = protein_peptide_merged['Peptide'] + '_' + protein_peptide_merged['UniProt'] + '_AbexNPX'
    protein_peptide_AbexNPX['AbexNPX'] = protein_peptide_AbexNPX['PeptideAbundance'] * protein_peptide_AbexNPX['NPX']


    PepAbundance = protein_peptide_Abe.pivot_table(index=['visit_id', 'visit_month'], columns='pep_prot_Abe', values='PeptideAbundance')
    PepAbundance = PepAbundance.sort_values(by='visit_id')
    
    PepNPX = protein_peptide_NPX.pivot_table(index=['visit_id', 'visit_month'], columns='pep_prot_NPX', values='NPX')
    PepNPX = PepNPX.sort_values(by='visit_id')

    PepAbundancexNPX = protein_peptide_AbexNPX.pivot_table(index=['visit_id', 'visit_month'], columns='pep_prot_AbexNPX', values='AbexNPX')
    PepAbundancexNPX = PepAbundancexNPX.sort_values(by='visit_id')

    merged1 = pd.merge(PepAbundance, PepNPX, on=['visit_id', 'visit_month'], how='left')
    all_merged = pd.merge(merged1, PepAbundancexNPX, on=['visit_id', 'visit_month'], how='left')

    all_merged = all_merged.fillna(0)
    train_column_names = X_train.columns.tolist()
    test_column_names = all_merged.columns.tolist()
    columns_not_in_test = list(set(train_column_names) - set(test_column_names))
    all_merged = all_merged.assign(**dict.fromkeys(columns_not_in_test, 0))
    all_merged = all_merged.drop(columns=['visit_month'])
    total_data = pd.merge(train_ids, all_merged, on='visit_id', how='right')
    return total_data


# list containing all the results for each iteration of test values
result_df_list = []

# funtion to make predictions and output them in a submission file
def get_predictions(test, test_proteins, test_peptides, sample_submission):
    results = {}
    test_features = get_features(test, test_proteins, test_peptides, unique_peptides_in_train)
    
    test_index = test_features[['visit_id', 'visit_month', 'patient_id']]
    test_data = test_features.drop(['visit_id', 'patient_id'], axis=1)
    test_data = test_data.sort_index(axis=1)

    predictions_needed=['updrs_1_updrs+0', 'updrs_1_updrs+6', 'updrs_1_updrs+12', 'updrs_1_updrs+18', 'updrs_1_updrs+24', 'updrs_1_updrs+30', 
                         'updrs_2_updrs+0', 'updrs_2_updrs+6', 'updrs_2_updrs+12', 'updrs_2_updrs+18', 'updrs_2_updrs+24', 'updrs_2_updrs+30',
                         'updrs_3_updrs+0', 'updrs_3_updrs+6', 'updrs_3_updrs+12', 'updrs_3_updrs+18', 'updrs_3_updrs+24', 'updrs_3_updrs+30']
    
    test_data = ss.transform(test_data)
    test_data = pca.transform(test_data)
    for pred in predictions_needed:
        results[pred] = models[pred].predict(test_data)
    

    results_df = pd.DataFrame(results)  

    indexed_result_df = pd.concat([test_index, results_df], axis = 1) 
    result_df_list.append(indexed_result_df)
    
    for idx, (prediction_id, rating) in sample_submission.iterrows():

        splt = prediction_id.split('_')
        
        target = splt[2] + '_' + splt[3] + '_updrs+' + splt[-2]
        updrs = int(splt[3])
        visit_month = int(splt[1])
        month_offset = int(splt[-2])
        visit_id = splt[0] + '_' + splt[1]
        patient_id = int(splt[0])
     
        
        if not any(patient_id in df['patient_id'].tolist() for df in result_df_list):
            continue
        elif updrs == 4:
            continue 
        else:
            for pred_df in result_df_list:
                if visit_id in pred_df['visit_id'].tolist() and target in predictions_needed:
                    sample_submission.loc[idx, 'rating'] = pred_df.loc[pred_df['visit_id'] == visit_id, target].iloc[0]
                elif patient_id in pred_df['patient_id'].tolist():
                    available_visit_month = pred_df.loc[pred_df['patient_id'] == patient_id, 'visit_month'].iloc[0]
                    month_offset = month_offset + (visit_month - available_visit_month)
                    target = splt[2] + '_' + splt[3] + '_updrs+' + str(month_offset)
                    if target in predictions_needed:
                        sample_submission.loc[idx, 'rating'] = pred_df.loc[pred_df['patient_id'] == patient_id, target].iloc[0]
    
    return sample_submission




# API for submission into kaggle (only works in kaggle)
# import sys
# sys.path.append('/kaggle/input/amp-parkinsons-disease-progression-prediction/')

# import amp_pd_peptide
# amp_pd_peptide.make_env.func_dict['__called__'] = False
# env = amp_pd_peptide.make_env()

# iter_test = env.iter_test() 

# for (test, test_peptides, test_proteins, sample_submission) in iter_test:
#     print(sample_submission)
#     result = get_predictions(test, test_proteins, test_peptides, sample_submission)
#     env.predict(sample_submission)
#     print(result)