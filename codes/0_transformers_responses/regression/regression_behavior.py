import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from utils.config import BASE_DIR, SEED, LAYERS_END, MODELS, MODELS, LAYERS_END
from utils.helpers import *
from utils.regression import compute_correlation
from utils.arg_parser import create_behavior_parser, parse_common_args
from utils.helpers import get_descriptors, set_seeds
from datetime import datetime
from pathlib import Path




def main():
    set_seeds(seed=SEED)
    parser = create_behavior_parser('chem_exploration')
    args = parser.parse_args()
    args = parse_common_args(args)
    
    model_name_path = args.model
    model_path = model_name_path.split('/')[0]
    model_name = model_name_path.split('/')[1]
    m = MODELS.index(model_name)
    participant_id = args.participant_id
    n_components = args.n_components
    n_fold = args.n_fold
    out_dir = args.out_dir
    z_score = args.z_score
    ds= args.ds
    embed_type=args.embed_type
    embed_cols = args.behavior_embeddings or get_descriptors(ds)
    run_id=args.run_id





    for layer in range(1, LAYERS_END[m] + 1):
        # Load behavior embeddings
        train_embeddings= []
        train_behaviors=[]
        test_embeddings=[]
        test_behaviors=[]
        test_cids_folds = []
        for i_fold in range(n_fold):
            print(i_fold,layer)
            
            train_cids, test_cids = load_fold_cids( n_fold, i_fold, ds)
         
            # print(test_cids.shape,"shape")
            
            train_behavior = load_behavior_embeddings(ds,train_cids,participant_id, embed_cols, group_by_cid=True)
            test_behavior = load_behavior_embeddings(ds,test_cids,participant_id, embed_cols, group_by_cid=True)
            train_embedding = load_model_embeddings( ds,model_name,train_cids,layer,embed_type=embed_type)
            test_embedding=load_model_embeddings( ds,model_name,test_cids,layer,embed_type=embed_type)

            train_embeddings.append(train_embedding)
            test_embeddings.append(test_embedding)
            train_behaviors.append(train_behavior)
            test_behaviors.append(test_behavior)
            test_cids_folds.append(list(test_cids))  

            # Prepare data for all folds
       

        out_base = Path(BASE_DIR) / f"{out_dir}_behaviormetrics_{run_id}_alphapertarget"
        out_base.mkdir(parents=True, exist_ok=True)
        out_file = out_base / f"metrics_model-{model_name}_ds-{ds}.csv"


        
        # Compute correlations
        metrics,preds_list = compute_correlation(
            train_embeddings, train_behaviors, test_embeddings, test_behaviors, test_cids_folds,
         n_components=n_components,z_score=z_score
        )
        metrics = metrics.assign(
            model=model_name,
            ds=ds,
            participant_id=participant_id,
            layer=layer,
            n_fold=n_fold,
            n_components=n_components,
            z_score=z_score,
            target=embed_cols,
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_id=os.environ.get("RUN_ID", "UNKNOWN")
            
        )
        write_header = not out_file.exists()
        metrics.to_csv(out_file, mode="a", index=False, header=write_header)
        print("****")



        # ---- PREDICTIONS (LLM-like CSV) ----
        preds_df = collect_predictions_rows(preds_list,
            test_cids_list=test_cids_folds,
            descriptors=embed_cols,
            participant_id=participant_id,
            model_name=model_name,
            ds=ds,
            layer=layer,
            n_fold=n_fold,
            n_components=n_components,
            z_score=z_score,
            run_id=os.environ.get("RUN_ID", "UNKNOWN"),
            # input_type_col="isomericsmiles",
            # smiles_lookup=smiles_lookup,
            # name_lookup=name_lookup,
        )

            # Save alongside your LLM outputs, with a clear name
        out_preds = Path(BASE_DIR) /"results"/"responses"/ "transformer_responses" / f"{ds}_odor_regression_scores_model-{model_name}_layer-{layer}_nfold-{n_fold}_ncomp-{n_components}_z-{int(bool(z_score))}.csv"
        out_preds.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(out_preds, index=False)
        print(f"Saved predictions: {out_preds}")
if __name__ == "__main__":
    main()



