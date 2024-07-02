import pandas as pd
import ast
from constants import chemical_features_r
def select_features(input_file):
    ds_alva = pd.read_csv(input_file)
    # print(gs_lf_alva.columns.values.tolist())

    nonStereoSMILE = list(map(lambda x: "nonStereoSMILES___" + x, chemical_features_r))
    # IsomericSMILES = list(map(lambda x: "IsomericSMILES___" + x, chemical_features_r))
    selected_features = nonStereoSMILE
    features= ['main_idx','nonStereoSMILES']+selected_features
    # print("cc1", ds_alva.columns.values.tolist())
    ds_alva= ds_alva.rename(columns={"Unnamed: 0":"main_idx"})
    # print("cc2", ds_alva.columns.values.tolist())
    ds_alva_selected = ds_alva[features]
    ds_alva_selected = ds_alva_selected.fillna(0)
    ds_alva_selected['embeddings'] = ds_alva_selected[selected_features].values.tolist()
    return ds_alva_selected

def create_alva_y(input_file_alva,input_file_data,file_name):

    gs_lf_pom = pd.read_csv(input_file_data)


    gs_lf_alva = select_features(input_file_alva)
    print(gs_lf_alva.columns.values.tolist())

    gs_lf_y = gs_lf_pom.copy()  # only keep y
    gs_lf_y.index.names = ['main_idx']
    del gs_lf_y['embeddings']
    gs_lf_alva = pd.merge(gs_lf_alva, gs_lf_y, on='main_idx')
    try:
        gs_lf_alva['y'] = gs_lf_alva['y'].apply(ast.literal_eval)
    except KeyError:
        pass

    gs_lf_alva.head(5)
    gs_lf_alva.to_csv(file_name)




