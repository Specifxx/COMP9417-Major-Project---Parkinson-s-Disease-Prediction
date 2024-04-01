"""
The below is code used to submit to Kaggle
 * WILL NOT RUN LOCALLY *

Credit to https://www.kaggle.com/code/gunesevitan/amp-pdpp-baseline
"""
# import amp_pd_peptide

# env = amp_pd_peptide.make_env()
# iter_test = env.iter_test() 

# for (test, test_peptides, test_proteins, sample_submission) in iter_test:
    
#     print(test_proteins)
    
#     for idx, row in test_proteins.iterrows():
#         if row['UniProt'] in feature_proteins:
#             print('Adding!')
#             protein_dict[row['UniProt']][row['visit_id']] = row['NPX']
    
#     sample_submission['patient_id'] = sample_submission.apply('prediction_id').str.split('_', expand=True)[0].astype(int)
#     sample_submission['current_visit_month'] = sample_submission.apply('prediction_id').str.split('_', expand=True)[1].astype(int)
#     sample_submission['visit_month_offset'] = sample_submission.apply('prediction_id').str.split('_', expand=True)[5].astype(int)
#     sample_submission['prediction_visit_month'] = sample_submission['current_visit_month'] + sample_submission['visit_month_offset'].astype(int)
#     sample_submission['updrs'] = sample_submission.apply('prediction_id').str.split('_', expand=True)[3].astype(int)

#     for updrs in range(1, 5):
#         updrs_idx = sample_submission['updrs'] == updrs
#         sample_submission.loc[updrs_idx, 'rating'] = sample_submission.loc[updrs_idx, 'prediction_visit_month'].map(target_visit_month_medians[f'updrs_{updrs}'])
        
#         missing_idx = sample_submission['rating'].isnull()
#         # Iterate over missing prediction rows after mapping the baselines
#         for idx, row in sample_submission[updrs_idx & missing_idx].iterrows():
#             # Find the closest visit_month value from the baselines table
#             target_visit_month_median_idx = np.argmin(np.abs(target_visit_month_medians.index - row['prediction_visit_month']))
#             # Write the closest visit_month value to the unseen visit_month
#             sample_submission.loc[idx, 'rating'] = target_visit_month_medians.iloc[target_visit_month_median_idx, updrs - 1]
            
#     for idx, row in sample_submission.iterrows():
        
#         prediction_id = row['prediction_id']

#         splt = prediction_id.split('_')
        
#         target = splt[2] + '_' + splt[3]
        
#         visit_id = splt[0] + '_' + splt[1]
#         print(visit_id)
        
#         target_models = models[target]
        
#         patient_id = splt[0]
#         visit_month = int(splt[1])
#         month_offset = int(splt[-2])
#         updrs = int(splt[3])        
        
#         if updrs == 1:
#             if visit_id in protein_dict[updrs_1_protein]:
#                 prot_pred = get_prot_updrs_1_predictions(pd.DataFrame([[visit_month, month_offset, protein_dict[updrs_1_protein][visit_id]]], columns=['visit_month', 'month_offset', updrs_1_protein]))
#                 sample_submission.loc[idx, 'rating'] = prot_pred[0]
#         if updrs == 2:
#             if visit_id in protein_dict[updrs_2_protein]:
#                 prot_pred = get_prot_updrs_2_predictions(pd.DataFrame([[visit_month, month_offset, protein_dict[updrs_2_protein][visit_id]]], columns=['visit_month', 'month_offset', updrs_2_protein]))
#                 sample_submission.loc[idx, 'rating'] = prot_pred[0]
#         elif updrs == 3:
#             if visit_id in protein_dict[updrs_3_protein]:
#                 prot_pred = get_prot_updrs_3_predictions(pd.DataFrame([[visit_month, month_offset, protein_dict[updrs_3_protein][visit_id]]], columns=['visit_month', 'month_offset', updrs_3_protein]))
#                 sample_submission.loc[idx, 'rating'] = prot_pred[0]
    
#     sample_submission = sample_submission.loc[:, ['prediction_id', 'rating']]
#     env.predict(sample_submission)