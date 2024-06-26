import torch
from fast_transformers.masking import LengthMask as LM
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np 
import deepchem as dc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
import numpy as np
# import os
from IPython.display import Image
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr  
from sklearn.metrics import r2_score
import itertools
import math
def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size


#Freche distance
# __all__ = ['frdist']
def _c(ca, i, j, p, q):

    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i]-q[j])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, p, q), np.linalg.norm(p[i]-q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, p, q), np.linalg.norm(p[i]-q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i-1, j, p, q),
                _c(ca, i-1, j-1, p, q),
                _c(ca, i, j-1, p, q)
            ),
            np.linalg.norm(p[i]-q[j])
            )
    else:
        ca[i, j] = float('inf')

    return ca[i, j]
def embed(model, smiles, tokenizer, batch_size=64):
    # print(len(model.blocks.layers))
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.blocks.layers[0].register_forward_hook(get_activation('0'))
    model.blocks.layers[1].register_forward_hook(get_activation('1'))
    model.blocks.layers[2].register_forward_hook(get_activation('2'))
    model.blocks.layers[3].register_forward_hook(get_activation('3'))
    model.blocks.layers[4].register_forward_hook(get_activation('4'))
    model.blocks.layers[5].register_forward_hook(get_activation('5'))
    model.blocks.layers[6].register_forward_hook(get_activation('6'))
    model.blocks.layers[7].register_forward_hook(get_activation('7'))
    model.blocks.layers[8].register_forward_hook(get_activation('8'))
    model.blocks.layers[9].register_forward_hook(get_activation('9'))
    model.blocks.layers[10].register_forward_hook(get_activation('10'))
    model.blocks.layers[11].register_forward_hook(get_activation('11'))
    model.eval()
    embeddings = []
    keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    activations_embeddings = [[],[],[],[],[],[],[],[],[],[],[],[]]
    
    for batch in batch_split(smiles, batch_size=batch_size):
        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
        idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
        with torch.no_grad():
            
            token_embeddings = model.blocks(model.tok_emb(torch.as_tensor(idx)), length_mask=LM(mask.sum(-1)))
            
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            embeddings.append(embedding.detach().cpu())
            
            for i,key in enumerate(keys):
                transformer_output= activation[key]
                input_mask_expanded = mask.unsqueeze(-1).expand(transformer_output.size()).float()
                sum_embeddings = torch.sum(transformer_output * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = sum_embeddings / sum_mask
                activations_embeddings[i].append(embedding.detach().cpu())
    return embeddings, activations_embeddings


def frdist(p, q):
    """
    Computes the discrete Fréchet distance between
    two curves. The Fréchet distance between two curves in a
    metric space is a measure of the similarity between the curves.
    The discrete Fréchet distance may be used for approximately computing
    the Fréchet distance between two arbitrary curves,
    as an alternative to using the exact Fréchet distance between a polygonal
    approximation of the curves or an approximation of this value.

    This is a Python 3.* implementation of the algorithm produced
    in Eiter, T. and Mannila, H., 1994. Computing discrete Fréchet distance.
    Tech. Report CD-TR 94/64, Information Systems Department, Technical
    University of Vienna.
    http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf

    Function dF(P, Q): real;
        input: polygonal curves P = (u1, . . . , up) and Q = (v1, . . . , vq).
        return: δdF (P, Q)
        ca : array [1..p, 1..q] of real;
        function c(i, j): real;
            begin
                if ca(i, j) > −1 then return ca(i, j)
                elsif i = 1 and j = 1 then ca(i, j) := d(u1, v1)
                elsif i > 1 and j = 1 then ca(i, j) := max{ c(i − 1, 1), d(ui, v1) }
                elsif i = 1 and j > 1 then ca(i, j) := max{ c(1, j − 1), d(u1, vj) }
                elsif i > 1 and j > 1 then ca(i, j) :=
                max{ min(c(i − 1, j), c(i − 1, j − 1), c(i, j − 1)), d(ui, vj ) }
                else ca(i, j) = ∞
                return ca(i, j);
            end; /* function c */

        begin
            for i = 1 to p do for j = 1 to q do ca(i, j) := −1.0;
            return c(p, q);
        end.

    Parameters
    ----------
    P : Input curve - two dimensional array of points
    Q : Input curve - two dimensional array of points

    Returns
    -------
    dist: float64
        The discrete Fréchet distance between curves `P` and `Q`.

    Examples
    --------
    >>> from frechetdist import frdist
    >>> P=[[1,1], [2,1], [2,2]]
    >>> Q=[[2,2], [0,1], [2,4]]
    >>> frdist(P,Q)
    >>> 2.0
    >>> P=[[1,1], [2,1], [2,2]]
    >>> Q=[[1,1], [2,1], [2,2]]
    >>> frdist(P,Q)
    >>> 0
    """
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')

    if len_p != len_q or len(p[0]) != len(q[0]):
        raise ValueError('Input curves do not have the same dimensions.')

    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)

    dist = _c(ca, len_p-1, len_q-1, p, q)
    return dist

def cosine_similarity_df(df,col_name):
    """
    cosine_similarity_df function calculated cosine_similarity  for df
    :param p1: df is a dataframe with 'Combined' column as a column which all the features are combined in a list.
    It does have one entry per CID. It also contains a 'CID' column 
    :return: a square dataframe which is pair-wise cosine similarity for each pair of cids.
    """
    df = df.dropna(subset=[col_name])
    df_cid_combined = df[['CID', col_name]]
    list_cid_combined = df_cid_combined[col_name].to_list()
    df_cosine_sim_matrix = cosine_similarity(list_cid_combined)
    df_cosine_sim_df = pd.DataFrame(df_cosine_sim_matrix, index=df_cid_combined['CID'], columns=df_cid_combined['CID'])
    df_cosine_sim_df = df_cosine_sim_df.reindex(sorted(df_cosine_sim_df.columns), axis=1)
    df_cosine_sim_df=df_cosine_sim_df.sort_index(ascending=True)
    return df_cosine_sim_df



def PCA_df(df, col_name,n_components=5):
    """
    PCA_df finction calculates applys PCA  for a da
    :param p1: df is a dataframe with 'Combined' column as a column which all the features are combined in a list.
    It does have one entry per CID. It also contains a 'CID' column 
    :return: a square dataframe which is pair-wise cosine similarity for each pair of cids.
    """
    df_cid_combined = df[['CID', col_name]]
    list_cid_combined = df_cid_combined[col_name].to_list()
    pca = PCA(n_components)
    reduced_cid_combined = pca.fit_transform(list_cid_combined)
    df_reduced_cid_combined= pd.DataFrame(reduced_cid_combined, index=df_cid_combined['CID'])
    df_reduced_cid_combined[col_name]=df_reduced_cid_combined.loc[:, 0:4].values.tolist()
    df_reduced_cid_combined=df_reduced_cid_combined.reset_index()
    return df_reduced_cid_combined


# def postproce_molembeddings(embeddings,index):
#     # molecules_embeddings_penultimate = torch.cat(embeddings)
#     # molecules_embeddings_penultimate=torch.cat(molecules_embeddings_penultimate,index)
#     # df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate, index=index)
#     # df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':'767'].values.tolist()
#     # df_molecules_embeddings=df_molecules_embeddings.reset_index()
#     # return(df_molecules_embeddings)
#     molecules_embeddings_penultimate = torch.cat(embeddings)
#     print("sizeeee",molecules_embeddings_penultimate.size())
#     if index.ndim>1:
#         columns_size= int(molecules_embeddings_penultimate.size()[1]+index.ndim)
        
#         molecules_embeddings_penultimate = torch.cat((molecules_embeddings_penultimate, torch.from_numpy( index.to_numpy())), dim=1)
#         df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate,columns=[str(i) for i in range(columns_size)]).reset_index()
#         print(df_molecules_embeddings.columns)
#         df_molecules_embeddings=df_molecules_embeddings.set_index([str(columns_size-2), str(columns_size-1)])
#         df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':str(int(columns_size-index.ndim-1))].values.tolist()

        
#     else:
#         df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate, index=index)
#         df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':'767'].values.tolist()
#     df_molecules_embeddings=df_molecules_embeddings.reset_index()
#     return df_molecules_embeddings




def postproce_molembeddings(embeddings,index):
    # molecules_embeddings_penultimate = torch.cat(embeddings)
    # molecules_embeddings_penultimate=torch.cat(molecules_embeddings_penultimate,index)
    # df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate, index=index)
    # df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':'767'].values.tolist()
    # df_molecules_embeddings=df_molecules_embeddings.reset_index()
    # return(df_molecules_embeddings)
    molecules_embeddings_penultimate = torch.cat(embeddings)
    columns_size= int(molecules_embeddings_penultimate.size()[1])
    # print("sizeeee",molecules_embeddings_penultimate.size())
    if index.ndim>1:
        
        # print("index", index)
        
        molecules_embeddings_penultimate = torch.cat((  torch.from_numpy( index.to_numpy()),molecules_embeddings_penultimate), dim=1)
        # print("mmmm",molecules_embeddings_penultimate[0:4,0:4])
        df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate,columns=['CID','subject']+[str(i) for i in range(columns_size)])
        # print("ddd",df_molecules_embeddings.columns.tolist())
        
        
        df_molecules_embeddings=df_molecules_embeddings.set_index(['CID','subject'])
        df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':str(columns_size-1)].values.tolist()

        
    else:
        # molecules_embeddings_penultimate = torch.cat((torch.from_numpy(index.to_numpy()).unsqueeze(1),molecules_embeddings_penultimate), dim=1)
        df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate,columns=[str(i) for i in range(columns_size)])
        df_molecules_embeddings['CID']=index
        df_molecules_embeddings=df_molecules_embeddings.set_index(['CID'])
        df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':str(columns_size-1)].values.tolist()
    df_molecules_embeddings=df_molecules_embeddings.reset_index()
    return df_molecules_embeddings






def sortet_pearson(df1,df2):
    df1 = df1.reindex(sorted(df1.columns), axis=1)
    df1=df1.sort_index(ascending=True)

    df2 = df2.reindex(sorted(df2.columns), axis=1)
    df2=df2.sort_index(ascending=True)
    
    return pearsonr(df1.to_numpy().flatten(), df2.to_numpy().flatten())


def cluster_tightness(in_group, out_group):
    """
    Compute the Cluster Tightness for an odor class.

    Parameters:
    - in_group: List of points in the in-group (odor class).
    - out_group: List of points in the out-group.

    Returns:
    - Cluster Tightness value.
    """
    mean_in_group_distance = mean_euclidean_distance(in_group,in_group)
    mean_in_out_group_distance = mean_euclidean_distance(in_group , out_group)
    
    if mean_in_out_group_distance == 0:
        return float('inf')  # Handle the case where the denominator is zero
    
    cluster_tightness_value = mean_in_group_distance / mean_in_out_group_distance
    return cluster_tightness_value


def get_cluster_tightness(X,y):
    in_group_data = {}
    out_group_data = {}
    for label_index in range(y.shape[1]):
        # Select data points with the current label
        current_label_data_indices_in = np.where(y[:, label_index] == 1)[0]
        current_label_data_indices_out = np.where(y[:, label_index] != 1)[0]
        in_group = X[current_label_data_indices_in]
        out_group = X[current_label_data_indices_out]
    
        # Split the data for in-group and out-group
        # in_group, out_group = train_test_split(current_label_data, test_size=0.2, random_state=42)
    
        in_group_data[label_index] = in_group
        out_group_data[label_index] = out_group
    
    # Example usage:
    results=[]
    for label_index in range(y.shape[1]):
        result = cluster_tightness(in_group_data[label_index], out_group_data[label_index])
        results.append(result)
        # print(f"Cluster Tightness for Label {label_index}: {result}")
    print(sum(results)/len(results))
    return results



#Added Apr 8
def run_class_time_CV_fmri_crossval_ridge(chemical_data,data,
                                          regress_feat_names_list = [],method = 'kernel_ridge', 
                                          lambdas = np.array([0.1,1,10,100,1000]),
                                          detrend = False, n_folds = 5, skip=5):
    
    # nlp_feat_type = predict_feat_dict['nlp_feat_type']
    # feat_dir = predict_feat_dict['nlp_feat_dir']
    # layer = predict_feat_dict['layer']
    # seq_len = predict_feat_dict['seq_len']
        
    
    n_odors = data.shape[0]     # (1211)
    n_voxels = data.shape[1]    # (~27905)

    ind = CV_ind(n_odors, n_folds=n_folds)      # (1211,) => [0., 0., 0., ..., 3., 3., 3.]


    corrs = np.zeros((n_folds, n_voxels))
    acc = np.zeros((n_folds, n_voxels))
    acc_std = np.zeros((n_folds, n_voxels))

    all_test_data = []
    all_preds = []
    
    
    for ind_num in range(n_folds):
        train_ind = ind!=ind_num                # (1211,) => [False, False, False, ...,  True,  True,  True]
        test_ind = ind==ind_num                 # (1211,) => [ True,  True,  True, ..., False, False, False]
        print(len(train_ind),len(test_ind))
        
        _,_,chemical_train_data,chemical_test_data = get_chemical_features_fixed_length(chemical_data, train_ind,test_ind)
        fmri_train_features,fmri_test_features = prepare_fmri_features(data,train_ind,test_ind)
        all_test_data.append(fmri_test_features)
        
        # start_time = tm.time()
        weights, chosen_lambdas = cross_val_ridge(chemical_train_data,fmri_train_features, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'plain',do_plot = False)

        preds = np.dot(chemical_test_data, weights)
        corrs[ind_num,:] = corr(preds,fmri_test_features)
        res=stats.pearsonr(preds.reshape(-1), fmri_test_features.reshape(-1))
        res1=np.corrcoef(preds.reshape(-1), fmri_test_features.reshape(-1))
        all_preds.append(preds)
            
        # print('fold {} completed, took {} seconds'.format(ind_num, tm.time()-start_time))
        del weights

    return corrs, acc, acc_std, np.vstack(all_preds), np.vstack(all_test_data),res.statistic,res1


def CV_ind(n, n_folds):
    ind = np.zeros((n))                         # (1211,)
    n_items = int(np.floor(n/n_folds))          # 302
    for i in range(0,n_folds -1):               # Folds 0,1,2
        ind[i*n_items:(i+1)*n_items] = i
    ind[(n_folds-1)*n_items:] = (n_folds-1)     # Fold 3
    return ind    



# def get_chemical_features_fixed_length(layer, seq_len, feat_type, feat_dir,  train_ind,test_ind):
def get_chemical_features_fixed_length(chemical_data,  train_ind,test_ind):
    
    # loaded = np.load( os.path.join(feat_dir, feat_type + '_length_'+str(seq_len)+ '_layer_' + str(layer) + '.npy') )
    # For now, all models should be processed the same way
    # In future, if there are newer models that are processed differently, can add additional if-branch 
    if True:
        train = chemical_data[train_ind]         # (~3877, 768)
        test = chemical_data[test_ind]         # (~1279, 768)
    else:
        print('Unrecognized NLP feature type {}.'.format(feat_type))
    
    pca = PCA(n_components=20, svd_solver='full')
    pca.fit(train)
    train_pca = pca.transform(train)                                    # (~3877, 10)
    test_pca = pca.transform(test)                                      # (~1279, 10)

    return train, test, train_pca, test_pca 


def prepare_fmri_features(tmp,train_ind,test_ind):
        
    return tmp[train_ind], tmp[test_ind]



def chemical_aggregator(x_list):
    means = []
    for x in x_list:
        if ',' in x:
            # print(x)
            int_list = [float(x_str) for x_str in x.split(',')]
        else:
            int_list = [float(x)]
        filtered = [col for col in int_list if col not in [-1]]
        if len(filtered)==0:
            return -1
        mean = sum(filtered) / len(filtered)
        means.append(mean)
    return means[0]
    
    
def pivot_helper(df,values):
    df_pivoted = df.pivot(index='CID Stimulus 1', columns='CID Stimulus 2', values=values)
    # df_ravia_mean_pivoted.head(5)
    df_pivoted = df_pivoted.reindex(sorted(df_pivoted.columns), axis=1)
    df_pivoted=df_pivoted.sort_index(ascending=True)
    return df_pivoted
# pivot_helper()



def add_chemical_features(df):
    chemical_features_r=["nCIR",
                             "ZM1", 
                             "GNar", 
                             "S1K", 
                             "piPC08",
                             "MATS1v",
                             "MATS7v",
                             "GATS1v", 
                             "Eig05_AEA(bo)", 
                             "SM02_AEA(bo)",
                             "SM03_AEA(dm)",
                             "SM10_AEA(dm)",
                             "SM13_AEA(dm)",
                              "SpMin3_Bh(v)",
                             "RDF035v",
                             "G1m",
                             "G1v",
                             "G1e",
                             "G3s",
                             "R8u+",
                             "nRCOSR"]
    
        
    nonStereoSMILE1 = list(map(lambda x: "Stimulus 1-nonStereoSMILES___" + x, chemical_features_r))
    nonStereoSMILE2 = list(map(lambda x: "Stimulus 2-nonStereoSMILES___" + x, chemical_features_r))
    IsomericSMILES1 = list(map(lambda x: "Stimulus 1-IsomericSMILES___" + x, chemical_features_r))
    IsomericSMILES2 = list(map(lambda x: "Stimulus 2-IsomericSMILES___" + x, chemical_features_r))
    
    for chemical_feature_r,feature_1,feature_2 in zip(chemical_features_r,nonStereoSMILE1,nonStereoSMILE2):
           
        df["diff-nonStereoSMILES___"+chemical_feature_r] = df[feature_1] - df[feature_2]
        df["diff-nonStereoSMILES___"+chemical_feature_r] = np.where(df[feature_1] == -1, -1, df["diff-nonStereoSMILES___"+chemical_feature_r])
        df["diff-nonStereoSMILES___"+chemical_feature_r] = np.where(df[feature_2] == -1, -1, df["diff-nonStereoSMILES___"+chemical_feature_r])
    
    for chemical_feature_r,feature_1,feature_2 in zip(chemical_features_r,IsomericSMILES1,IsomericSMILES2):
    
        df["diff-IsomericSMILES___"+chemical_feature_r] = df[feature_1] - df[feature_2]
        df["diff-IsomericSMILES___"+chemical_feature_r] = np.where(df[feature_1] == -1, -1, df["diff-IsomericSMILES___"+chemical_feature_r])
        df["diff-IsomericSMILES___"+chemical_feature_r] = np.where(df[feature_2] == -1, -1, df["diff-IsomericSMILES___"+chemical_feature_r])
        
    return df


# def prepare_mols_helper(modeldeepchem,df_mols,mol_type="nonStereoSMILES",index="CID"):
#     df_mols_layers=[]
#     df_mols_layers_zscored=[]
    
#     #inference on molecules
#     df_mols_embeddings_original, df_mols_layers_original=embed(lm,df_mols[mol_type], tokenizer, batch_size=64)
    
    
        
#     df_mols_embeddings=postproce_molembeddings(df_mols_embeddings_original,df_mols[index])
#     print("columns",df_mols_embeddings.columns)

    
    
#     df_mols_embeddings_diskdataset = dc.data.DiskDataset.from_numpy(df_mols_embeddings['Combined'].values.tolist())
#     df_mols_embeddings_linear=modeldeepchem.predict_embedding(df_mols_embeddings_diskdataset)
#     df_mols_embeddings_linear_torch=[torch.from_numpy(x.reshape(1,-1)) for x in df_mols_embeddings_linear]
#     df_mols_embeddings_linear=postproce_molembeddings(df_mols_embeddings_linear_torch,df_mols[index])
    
    
#      #z-score embeddings
#     df_mols_embeddings_zscored = df_mols_embeddings.copy()
#     scaled_features = StandardScaler().fit_transform(df_mols_embeddings_zscored.loc[:, '0':'767'].values.tolist())
#     df_mols_embeddings_zscored.loc[:, '0':'767'] = pd.DataFrame(scaled_features, index=df_mols_embeddings_zscored.index, columns=[str(i) for i in range(768)])
#     df_mols_embeddings_zscored['Combined'] = df_mols_embeddings_zscored.loc[:, '0':'767'].values.tolist()
    
    
    
#     #z-score linear embeddings
#     df_mols_embeddings_linear_zscored = df_mols_embeddings_linear.copy()
#     scaled_features = StandardScaler().fit_transform(df_mols_embeddings_linear_zscored.loc[:, '0':'255'].values.tolist())
#     df_mols_embeddings_linear_zscored.loc[:, '0':'255'] = pd.DataFrame(scaled_features, index=df_mols_embeddings_linear_zscored.index, columns=[str(i) for i in range(256)])
#     df_mols_embeddings_linear_zscored['Combined'] = df_mols_embeddings_linear_zscored.loc[:, '0':'255'].values.tolist()


    
#     for df_mols_layer in df_mols_layers_original:
#         df_mols_layer=postproce_molembeddings(df_mols_layer,df_mols[index])
#         df_mols_layers.append(df_mols_layer)
#         # print("step2")
        
#          #z-score embeddings
#         df_mols_embeddings_zscored = df_mols_layer.copy()
#         scaled_features = StandardScaler().fit_transform(df_mols_embeddings_zscored.loc[:, '0':'767'].values.tolist())
#         df_mols_embeddings_zscored.loc[:, '0':'767'] = pd.DataFrame(scaled_features, index=df_mols_embeddings_zscored.index, columns=[str(i) for i in range(768)])
#         df_mols_embeddings_zscored['Combined'] = df_mols_embeddings_zscored.loc[:, '0':'767'].values.tolist()
#         df_mols_layers_zscored.append(df_mols_embeddings_zscored)
        
    
#     return df_mols_embeddings_original,df_mols_layers_original,df_mols_embeddings,df_mols_embeddings_zscored,df_mols_layers,df_mols_layers_zscored,df_mols_embeddings_linear,df_mols_embeddings_linear_zscored


def cosine_sim_helper_layers(df_mols_embeddings, df_mols_embeddings_zscored, df_mols_layers, df_mols_layers_zscored):

    # cosine_sim_df_mols_embeddings=cosine_similarity_df(df_mols_embeddings,'Combined')
    # cosine_sim_df_mols_embeddings_zscored=cosine_similarity_df(df_mols_embeddings_zscored,'Combined')


    ### Cosine similarity for all layers
    cosine_sim_df_mols_layers = []
    cosine_sim_df_mols_layers_zscored = []
    # embedding = molecules_activations_embeddings_original[0]
    for embeddings in df_mols_layers:
        cosine_sim_df_mols_layers.append(cosine_similarity_df(embeddings,'Combined'))

    for embeddings in df_mols_layers_zscored:
        cosine_sim_df_mols_layers_zscored.append(cosine_similarity_df(embeddings,'Combined'))

    return  cosine_sim_df_mols_layers, cosine_sim_df_mols_layers_zscored


def cosine_sim_helper(df_mols_embeddings, df_mols_embeddings_zscored):
    
    cosine_sim_df_mols_embeddings=cosine_similarity_df(df_mols_embeddings,'Combined')
    cosine_sim_df_mols_embeddings_zscored=cosine_similarity_df(df_mols_embeddings_zscored,'Combined')
    return cosine_sim_df_mols_embeddings, cosine_sim_df_mols_embeddings_zscored
    
def correlation_helper_layers(df_all,cosine_sim_df_mols_layers_zscored,equalize_size=True,value_type="r"):
    layers=[]
    layers_pvalue=[]

    for i in range(len(cosine_sim_df_mols_layers_zscored)):
        out_mols=cosine_sim_df_mols_layers_zscored[i]
        data_flattered=flattening_data_helper(df_all,out_mols,equalize_size=True)
        if value_type=="R2":
            result=r2_score(data_flattered["Peceptual Similarity"], data_flattered["Model Similarity"])
            layers.append(result)
        else:
            result=pearsonr(data_flattered["Peceptual Similarity"], data_flattered["Model Similarity"])
            layers.append(result.statistic)
            layers_pvalue.append(result.pvalue)
    if value_type=="R2":
        return layers
    else:    
        return layers,layers_pvalue


def correlation_helper_mixture(df_all,df_mols_all,value_type="r"):
    # layers=[]
    # layers_pvalue=[]
    data_flattered=flattening_data_helper(df_all,df_mols_all,equalize_size=True)
    
    
    if value_type=="R2":
        last=r2_score(data_flattered["Peceptual Similarity"], data_flattered["Model Similarity"])
    else:       
        last=pearsonr(data_flattered["Peceptual Similarity"], data_flattered["Model Similarity"])
    if value_type=="R2":
        return last
    else:    
        return last.statistic,last.pvalue


def flattening_data_helper(out_original,out_mols,equalize_size=True):

    out_mols=out_mols.to_numpy().flatten()
    out=out_original.to_numpy().flatten()
    if equalize_size:
        to_be_deleted=np.argwhere(out!=out).flatten().tolist()
        out=np.delete(out,to_be_deleted)
        out_mols=np.delete(out_mols,to_be_deleted)
        
    data = {"Peceptual Similarity": out, "Model Similarity": out_mols}
    return data








def plot_lines(data,title,filename):
    df_corrs=pd.DataFrame.from_dict(data, orient='index',
                       columns=['Dataset', 'Correlation'])
    df_corrs=df_corrs.explode('Correlation')
    df_corrs['Layer'] = df_corrs.groupby(level=0).cumcount()
    #alternative
    #df['idx'] = df.groupby(df.index).cumcount()
    df_corrs = df_corrs.reset_index(drop=True)
    sns.set_style("white")
    # df[['0', '1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11']] = df['All layers'].str.split(' ', expand=True)
    fig, ax = plt.subplots(figsize=(7.5, 7.5)
                           # ,constrained_layout = True
                          )
    
    
    # sns.color_palette("tab10")
    sns.color_palette("hls", 4)
    palette=[ "#56c4c0",
                "#f2c800",
                 # "#d73027",
                 # "#fc8d59",
                 # "#ffda33",
                
                # , "#4575b4"
    ]
    g=sns.lineplot(data=df_corrs, x="Layer", y="Correlation",hue="Dataset",palette = palette, lw=7)
    # sns.barplot(df_corrs, x="Dataset", y="Correlation", hue= "Correlation",width=0.2,legend=False,palette=sns.color_palette("Set2",4))
    
    
    ax.legend().set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
     # axes = plt.gca() #Getting the current axis
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #56c4c0fb
    
    fig.subplots_adjust(bottom=0.32,left=0.2)
    fig.legend(handles, labels, ncol=1, columnspacing=1, prop={'size': 25}, handlelength=1.5, loc="lower center",
               borderpad=0.3,
               bbox_to_anchor=(0.54, 0),
               
               frameon=True, labelspacing=0.4,handletextpad=0.2)
    g.set_xticks([0,2,4,6,8,10]) # <--- set the ticks first
    g.set_xticklabels(['1', '3', '5', '7', '9', '11'])
    
    # g.set_yticks([0.45,0.5,0.55,0.6,0.65,0.7]) # <--- set the ticks first
    # g.set_yticklabels(['', '0.5','', '0.6','', '0.7'])
    
    # g.set_yticks([0.45,0.5,0.55,0.6,0.65,0.7]) # <--- set the ticks first
    # g.set_yticklabels(['', '0.5','', '0.6','', '0.7'])
    # g.set_ylim(0.45,0.7)
    
    g.set_xlim(0,11)
    ax.set_ylabel('')
    ax.set_xlabel('Model Layer')
    # plt.margins(0,-0.16)
    ax.xaxis.set_label_coords(0.5, -0.13)
    
    # plt.tight_layout()
    plt.savefig(filename
                # , bbox_inches="tight"
               
               )


# def extract_embedding_molformer(lm,tokenizer,Tasks,input_file,smiles_field):
#
#     featurizer = dc.feat.DummyFeaturizer()
#
#     randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
#     loader = dc.data.CSVLoader(tasks=Tasks,
#                        feature_field=smiles_field,
#                        featurizer=featurizer
#                               )
#     dataset = loader.create_dataset(inputs=[input_file])
#     n_tasks = len(dataset.tasks)
#     train_dataset, test_dataset, valid_dataset = randomstratifiedsplitter.train_valid_test_split(dataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1, seed = 1)
#     train_embeddings_original, train_activations_embeddings_original=embed(lm, train_dataset.X, tokenizer, batch_size=64)
#     test_embeddings_original, test_activations_embeddings_original=embed(lm, test_dataset.X, tokenizer, batch_size=64)
#     valid_embeddings_original, valid_activations_embeddings_original=embed(lm, valid_dataset.X, tokenizer, batch_size=64)
#     embeddings_original, activations_embeddings_original=embed(lm, dataset.X, tokenizer, batch_size=64)
#
#
#
#
#
#
#
#
#     embedding_train_dataset,embedding_train_dataset_zscored=convert_embeddings(train_embeddings_original,train_dataset.y)
#     embedding_test_dataset,embedding_test_dataset_zscored=convert_embeddings(test_embeddings_original,test_dataset.y)
#     embedding_valid_dataset,embedding_valid_dataset_zscored=convert_embeddings(valid_embeddings_original,valid_dataset.y)
#     embedding_dataset,embedding_dataset_zscored=convert_embeddings(embeddings_original,dataset.y)
#
#
#     train_layers=[]
#     train_layers_zscored=[]
#     test_layers=[]
#     test_layers_zscored=[]
#     valid_layers=[]
#     valid_layers_zscored=[]
#     dataset_layers=[]
#     dataset_layers_zscored=[]
#
#     for df_mols_layer in train_activations_embeddings_original:
#         embedding_train_dataset_layer ,embedding_train_dataset_zscored_layer= convert_embeddings(df_mols_layer,train_dataset.y)
#         train_layers.append(embedding_train_dataset_layer)
#         train_layers_zscored.append(embedding_train_dataset_zscored_layer)
#
#     for df_mols_layer in test_activations_embeddings_original:
#         embedding_test_dataset_layer,embedding_test_dataset_zscored_layer =convert_embeddings(df_mols_layer,test_dataset.y)
#         test_layers.append(embedding_test_dataset_layer)
#         test_layers_zscored.append(embedding_test_dataset_zscored_layer)
#
#     for df_mols_layer in valid_activations_embeddings_original:
#         embedding_valid_dataset_layer,embedding_valid_dataset_zscored_layer=convert_embeddings(df_mols_layer,valid_dataset.y)
#         valid_layers.append(embedding_valid_dataset_layer)
#         valid_layers_zscored.append(embedding_valid_dataset_zscored_layer)
#
#     for df_mols_layer in activations_embeddings_original:
#         embedding_dataset_layer,embedding_dataset_zscored_layer=convert_embeddings(df_mols_layer,dataset.y)
#         dataset_layers.append(embedding_dataset_layer)
#         dataset_layers_zscored.append(embedding_dataset_zscored_layer)
#
#
#
#     return dataset, embedding_train_dataset,embedding_test_dataset,embedding_valid_dataset,embedding_dataset,train_layers,test_layers,valid_layers,dataset_layers,\
#     embedding_train_dataset_zscored,embedding_test_dataset_zscored,embedding_valid_dataset_zscored,embedding_dataset_zscored,train_layers_zscored,test_layers_zscored,valid_layers_zscored,dataset_layers_zscored
#






# def extract_embedding_molformer_fromfile(lm,tokenizer,Tasks,input_file,smiles_field):
#
#     featurizer = dc.feat.DummyFeaturizer()
#
#     randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
#     loader = dc.data.CSVLoader(tasks=Tasks,
#                        feature_field=smiles_field,
#                        featurizer=featurizer
#                               )
#     dataset = loader.create_dataset(inputs=[input_file])
#     n_tasks = len(dataset.tasks)
#     train_dataset, test_dataset, valid_dataset = randomstratifiedsplitter.train_valid_test_split(dataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1, seed = 1)
#     train_embeddings_original, train_activations_embeddings_original=embed(lm, train_dataset.X, tokenizer, batch_size=64)
#     test_embeddings_original, test_activations_embeddings_original=embed(lm, test_dataset.X, tokenizer, batch_size=64)
#     valid_embeddings_original, valid_activations_embeddings_original=embed(lm, valid_dataset.X, tokenizer, batch_size=64)
#     embeddings_original, activations_embeddings_original=embed(lm, dataset.X, tokenizer, batch_size=64)
#
#
#
#
#
#
#
#
#     embedding_train_dataset,embedding_train_dataset_zscored=convert_embeddings(train_embeddings_original,train_dataset.y)
#     embedding_test_dataset,embedding_test_dataset_zscored=convert_embeddings(test_embeddings_original,test_dataset.y)
#     embedding_valid_dataset,embedding_valid_dataset_zscored=convert_embeddings(valid_embeddings_original,valid_dataset.y)
#     embedding_dataset,embedding_dataset_zscored=convert_embeddings(embeddings_original,dataset.y)
#
#
#     train_layers=[]
#     train_layers_zscored=[]
#     test_layers=[]
#     test_layers_zscored=[]
#     valid_layers=[]
#     valid_layers_zscored=[]
#     dataset_layers=[]
#     dataset_layers_zscored=[]
#
#     for df_mols_layer in train_activations_embeddings_original:
#         embedding_train_dataset_layer ,embedding_train_dataset_zscored_layer= convert_embeddings(df_mols_layer,train_dataset.y)
#         train_layers.append(embedding_train_dataset_layer)
#         train_layers_zscored.append(embedding_train_dataset_zscored_layer)
#
#     for df_mols_layer in test_activations_embeddings_original:
#         embedding_test_dataset_layer,embedding_test_dataset_zscored_layer =convert_embeddings(df_mols_layer,test_dataset.y)
#         test_layers.append(embedding_test_dataset_layer)
#         test_layers_zscored.append(embedding_test_dataset_zscored_layer)
#
#     for df_mols_layer in valid_activations_embeddings_original:
#         embedding_valid_dataset_layer,embedding_valid_dataset_zscored_layer=convert_embeddings(df_mols_layer,valid_dataset.y)
#         valid_layers.append(embedding_valid_dataset_layer)
#         valid_layers_zscored.append(embedding_valid_dataset_zscored_layer)
#
#     for df_mols_layer in activations_embeddings_original:
#         embedding_dataset_layer,embedding_dataset_zscored_layer=convert_embeddings(df_mols_layer,dataset.y)
#         dataset_layers.append(embedding_dataset_layer)
#         dataset_layers_zscored.append(embedding_dataset_zscored_layer)
#
#
#
#     return dataset, embedding_train_dataset,embedding_test_dataset,embedding_valid_dataset,embedding_dataset,train_layers,test_layers,valid_layers,dataset_layers,\
#     embedding_train_dataset_zscored,embedding_test_dataset_zscored,embedding_valid_dataset_zscored,embedding_dataset_zscored,train_layers_zscored,test_layers_zscored,valid_layers_zscored,dataset_layers_zscored







# def extract_embedding_molformer_brief(lm,tokenizer,Tasks,input_file,smiles_field):
#
#     featurizer = dc.feat.DummyFeaturizer()
#
#     randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
#     loader = dc.data.CSVLoader(tasks=Tasks,
#                        feature_field=smiles_field,
#                        featurizer=featurizer
#                               )
#     dataset = loader.create_dataset(inputs=[input_file])
#     n_tasks = len(dataset.tasks)
#
#     embeddings_original, activations_embeddings_original=embed(lm, dataset.X, tokenizer, batch_size=64)
#
#
#
#
#
#     # print(dataset.y)
#
#     embeddings_original = torch.cat(embeddings_original).numpy()
#     X=torch.from_numpy(embeddings_original)
#     if len(Tasks)!=0:
#         y=dataset.y
#     else:
#         y=None
#
#
#     X_layers=[]
#     y_layers=[]
#     # dataset_layers_zscored=[]
#
#
#
#
#
#     for df_mols_layer in activations_embeddings_original:
#
#         embeddings_original = torch.cat(df_mols_layer).numpy()
#         X=torch.from_numpy(embeddings_original)
#         X_layers.append(X)
#         if len(Tasks)!=0:
#             # y=torch.from_numpy(y)
#             y_layers.append(y)
#         else:
#             y_layers.append(None)
#
#
#
#
#
#
#
#
#     return X,y,X_layers,y_layers









# def convert_embeddings(embeddings_original,y):
#     # print(y)
#     embeddings_original = torch.cat(embeddings_original).numpy()
#     y=torch.from_numpy(y)
#     embeddings=torch.from_numpy(embeddings_original)
#     embedding_dataset = torch.utils.data.TensorDataset(embeddings.cpu(), y)
#     embedding_loader = torch.utils.data.DataLoader(embedding_dataset, batch_size=128, shuffle=True)
#     embedding_dataset = dc.data.DiskDataset.from_numpy(embeddings_original.tolist(),y.tolist())
#
#
#     embeddings_zscored = embeddings.cpu()
#     embeddings_zscored = StandardScaler().fit_transform(embeddings_zscored)
#     embedding_dataset_zscored = dc.data.DiskDataset.from_numpy(embeddings_zscored,y.tolist())
#
#
#     return embedding_dataset,embedding_dataset_zscored

def compute_statistics(df_ravia_similarity_mols):
    ravia_mean=df_ravia_similarity_mols['nonStereoSMILES'].apply(len).mean()
    ravia_std=df_ravia_similarity_mols['nonStereoSMILES'].apply(len).std()
    ravia_max=df_ravia_similarity_mols['nonStereoSMILES'].apply(len).max()
    ravia_min=df_ravia_similarity_mols['nonStereoSMILES'].apply(len).min()
    print(ravia_min, ravia_max, ravia_mean, ravia_std)

    # def convert_embeddings_brief(embeddings_original,y):
    #
    #     return embeddings,y

