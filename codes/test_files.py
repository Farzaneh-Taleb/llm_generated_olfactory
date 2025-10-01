
import pandas as pd
ds = 'sagar2023'
INPUT_CSV = f'datasets/{ds}/{ds}_data.csv'
sagar = pd.read_csv(INPUT_CSV)

#count '' for isomericsmiles per participantt_id
sagar['isomericsmiles'].replace('', pd.NA, inplace=True)
missing_smiles = sagar[sagar['isomericsmiles'].isna()]
print(missing_smiles['participant_id'].value_counts())