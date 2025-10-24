
import argparse
import os
def create_behavior_parser(description='chem_exploration'):
    """
    Create argument parser for fMRI-specific scripts.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        ArgumentParser: Configured parser with fMRI-specific arguments
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Common arguments
    parser.add_argument('--participant_id', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--ds', type=str, required=True)
    parser.add_argument('--n_components', type=str, default="None")

    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--n_fold', type=int, required=True)
    parser.add_argument('--z_score', type=str, default="false")
    parser.add_argument("--run_id", default=os.environ.get("RUN_ID", "unknown"))
    parser.add_argument('embed_type', type=str, choices=['can', 'iso'], required=True, help="Type of embeddings: 'can' for canonical, 'iso' for isomeric")
    
    # fMRI-specific arguments
    
    
    return parser


def parse_common_args(args):
    """
    Parse and convert common arguments to appropriate types.
    
    Args:
        args: Parsed arguments object
        
    Returns:
        Parsed and converted arguments
    """
    # Convert string boolean arguments
    args.z_score = str(args.z_score).lower() == 'true'
    
    # Convert n_components
    if args.n_components == "None":
        args.n_components = None
    else:
        args.n_components = float(args.n_components)
    
    return args