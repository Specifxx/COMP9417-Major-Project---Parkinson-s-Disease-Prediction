# Group 'import pandas as np'

https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction

### Dependencies
- `environment.yaml` contains all package dependancies for the project
- You may need to modify the prefix line to your own environment directory
- The packages used should be found in most ML environments


### EDA.py
```
python3 EDA.py
```
- Contains the code used to generate our EDA graphics
- Generated graphics will be in the /output folder
- Note that the `missingno` package is not updated in Anaconda
    - It must be installed using PIP, otherwise the NaN values graph will not generate

### baseline_models.py
```
python3 baseline_models.py
```
- Contains our baseline models trained only on time
- Outputs the validation scores to stdout

### lgbm_model.py
```
python3 lgbm_model.py
```
- Contains our LGBM model train on time
- Uses a GridSearch to find optimal parameters
    - **Takes a long time to run**
    - RandomizedSearch can be used to save time
        - Pick one by uncommenting the chosen method
- Outputs the validation scores to stdout

### PCA_model.py
```
python3 PCA_model.py
```
- Contains our PCA model train on all Proteins and Peptides
- Takes approx 15 minutes to run
- Outputs the validation scores to stdout
    - List of scores for each target

### top_proteins_model.py
```
python3 top_proteins_model.py
```
- Contains our best model trained on handpicked proteins
    - The best proteins are found in the EDA output
- Two options for the supplementary model where Protein readings aren't available:
    - Rolling max median model
    - LGBM time model
    - Choose the supplementary model by uncommenting it on line 344/345
- Outputs the validation scores to stdout

### kaggle_boilerplate.py
- Contains boilerplate code for Kaggle submission
- Add to the end of top_proteins_model.py if you want to submit to Kaggle
- Will not run locally
