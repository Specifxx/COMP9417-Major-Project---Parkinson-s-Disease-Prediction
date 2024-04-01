import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from collections import defaultdict
import warnings



DATA_DIR = "supplied-files"
OUTPUT_DIR = "output"

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

train_clinical_data = pd.read_csv(DATA_DIR + "/train_clinical_data.csv")
train_peptides = pd.read_csv(DATA_DIR + "/train_peptides.csv")
train_proteins = pd.read_csv(DATA_DIR + "/train_proteins.csv")
supplemental_clinical_data = pd.read_csv(DATA_DIR + "/supplemental_clinical_data.csv")

print("\n============================== CLINICAL DATA ==================================")

print(train_clinical_data)
print(train_clinical_data.describe())

print("\n============================== PROTEIN DATA ==================================")

print(train_proteins)
print(train_proteins.describe())

print("\n============================== PEPTIDE DATA ==================================")

print(train_peptides)
print(train_peptides.describe())

print("\n============================== SUPPLEMENTAL DATA ==================================")


print(supplemental_clinical_data)
print(supplemental_clinical_data.describe())

#* NA Values
missingno.matrix(train_clinical_data)
plt.savefig(OUTPUT_DIR + '/na_values.png')

#* UPDRs vs visit_month
targets = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']
medians = train_clinical_data.groupby('visit_month',as_index=True)[targets].median()

# Strip Plot
fig, ax = plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True, figsize=(10,8))

for i, ax in enumerate(ax.flat):
    sns.stripplot(data = train_clinical_data, x='visit_month', y=f'updrs_{i+1}', jitter=2, size=1.5, alpha=.6, color='gray', ax=ax)
    sns.stripplot(data = medians.reset_index(), x='visit_month', y=f'updrs_{i+1}', color='green', ax=ax)
    ax.set_title(targets[i])
plt.suptitle('UPDRs and Medians vs visit_month')
plt.savefig(OUTPUT_DIR + '/updrs_visit_month_strip.png')

# Box Plot
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 25))
axs = ax.flatten()

for idx, target in enumerate(targets):
    sns.boxplot(data=train_clinical_data, x="visit_month", y=target, ax=axs[idx])
    
plt.savefig(OUTPUT_DIR + '/updrs_visit_month_box.png')

#* UPDRs Correlation Matrix
fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
ax = sns.heatmap(
    train_clinical_data[targets].corr(method="pearson"),
    annot=True,
    square=True,
    cmap='Reds',
    annot_kws={'size': 20},
)

plt.savefig(OUTPUT_DIR + '/updrs_corr.png')

#* On/Off/Unknown Medication
#checking if meds have any effect on each updrs
on_meds = train_clinical_data[train_clinical_data['upd23b_clinical_state_on_medication'] == 'On']
off_meds = train_clinical_data[train_clinical_data['upd23b_clinical_state_on_medication'] == 'Off']
na_meds = train_clinical_data[train_clinical_data['upd23b_clinical_state_on_medication'].isna()]

print("\n============================== CLINICAL DATA VS MEDICATION STATE ==================================")
print("\nON MEDS:")
print(on_meds.describe())
print("\nOFF MEDS:")
print(off_meds.describe())
print("\nUNKOWN:")
print(na_meds.describe())

on_meds.hist(figsize=(11, 7), bins=10)
plt.savefig(OUTPUT_DIR + '//on_meds')
off_meds.hist(figsize=(11, 7), bins=10)
plt.savefig(OUTPUT_DIR + '//off_meds')
na_meds.hist(figsize=(11, 7), bins=10)
plt.savefig(OUTPUT_DIR + '//na_meds')

#* Create Protein Dict
proteins = set()
protein_dict = defaultdict(dict)
for idx, row in train_proteins.iterrows():
    protein = row["UniProt"]
    proteins.add(protein)
    protein_dict[protein][row["visit_id"]] = row["NPX"]
    
prot_analysis_data = train_clinical_data.copy()
for protein in proteins:
    prot_analysis_data[protein] = prot_analysis_data["visit_id"].apply(lambda visit_id: np.nan if visit_id not in protein_dict[protein] else protein_dict[protein][visit_id])


#* Get Highest-Correlated Proteins
# spearman correlation as we cannot assume linear relationship
features = targets
features.extend(proteins)

prot_corr_matrix = prot_analysis_data[features].corr(method='spearman').abs()

sorted_corr = prot_corr_matrix.iloc[0:4, 4:].unstack().sort_values(ascending=False).drop_duplicates()

print("\n============================== HIGHEST CORRELATED PROTEINS ==================================")
print(sorted_corr.head(20))

#* Protein to UPDRs Correlation Matrix
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(20, 20), dpi=100)
axs = ax.flatten()

sns.heatmap(
    prot_corr_matrix.iloc[0:4, 4:34],
    annot=True,
    annot_kws={'size': 8},
    square=True,
    cmap='Reds',
    ax=ax[0],
    vmin=0,
    vmax=0.5,
)

sns.heatmap(
    prot_corr_matrix.iloc[0:4, 34:64],
    annot=True,
    annot_kws={'size': 8},
    square=True,
    cmap='Reds',
    ax=ax[1],
    vmin=0,
    vmax=0.5,
)

sns.heatmap(
    prot_corr_matrix.iloc[0:4, 64:94],
    annot=True,
    annot_kws={'size': 8},
    square=True,
    cmap='Reds',
    ax=ax[2],
    vmin=0,
    vmax=0.5,
)

sns.heatmap(
    prot_corr_matrix.iloc[0:4, 94:124],
    annot=True,
    annot_kws={'size': 8},
    square=True,
    cmap='Reds',
    ax=ax[3],
    vmin=0,
    vmax=0.5,
)

plt.savefig(OUTPUT_DIR + '/protein_corr.png')

#* Protein Appearance Frequency
protein_dict=defaultdict(dict)

for idx, (visit_id, visit_month, patient_id, UniProt, NPX) in train_proteins.iterrows():
    protein_dict[UniProt][visit_id] = NPX

unique_visits = train_clinical_data['visit_id'].unique()
    
protein_freq = {}
for k, v in sorted(protein_dict.items(), key=lambda x: len(x[1]), reverse=True):
    protein_freq[k] = len(v) * 1.0 / len(unique_visits) * 100
    
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 40))

# sns.barplot(protein_freq, ax=ax)
sns.barplot(y=list(protein_freq.keys()), x=list(protein_freq.values()), ax=ax)

ax.set_xlim([0, 100])
ax.set_title("% of Visits Containing Proteins")
ax.set_ylabel("Protein")
ax.set_xlabel("Visits (%)")
plt.savefig(OUTPUT_DIR + '/protein_frequency.png')



#* MODEL SCORES
# Values manually inputted from other outputs
baseline_model_scores = {
    'LGBM': 62.45,
    'Linear': 61.92,
    'XGB': 61.68,
    'LGBM tuned': 61.21,
}

fig, ax = plt.subplots()

ax.bar(
    [model for model in baseline_model_scores],
    [baseline_model_scores[model] for model in baseline_model_scores],
    color='tab:orange'
)

ax.set_ylim([50, 70])
ax.set_ylabel('SMAPE')
ax.set_title('Baseline Models SMAPE')

plt.savefig(OUTPUT_DIR + '/baseline_models.png')

protein_model_scores = {
    'AllProteins (AP)': 67.9,
    'TopProteins (TP)': 57.8,
    'AP + medians': 56.5,
    'TP + medians': 53.76
}

fig, ax = plt.subplots()

ax.bar(
    [model for model in protein_model_scores],
    [protein_model_scores[model] for model in protein_model_scores],
    color='tab:purple'
)

ax.set_ylim([50, 70])
ax.set_ylabel('SMAPE')
ax.set_title('Protein Models SMAPE')

plt.savefig(OUTPUT_DIR + '/protein_models.png')