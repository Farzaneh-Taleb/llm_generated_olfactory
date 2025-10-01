
#get smiles using molecule_id of pubchem
def get_smiles(molecule_id):
    pass

#get selfies
def get_selfies(molecule_id):
    pass

#get morgan fingerprints
def get_morgan_fingerprints(molecule_id, radius=2, nBits=2048):
    pass

def get_moldescriptors(molecule_id):
    """Get molecular descriptors for a given molecule ID."""
    pass

def get_ds_description(ds):
    pass

def get_ratings(ds,molecule_id,subject_id=None):
    pass


#classfication task with LLM
def classify_molecule(molecule_id,modality='smiles', model="gpt-4.1"):
    #choose the right funvtion based on modality
    if modality == 'smiles':
        smiles = get_smiles(molecule_id)
    elif modality == 'selfies':
        selfies = get_selfies(molecule_id)
    elif modality == 'morgan':
        fingerprints = get_morgan_fingerprints(molecule_id)
    else:
        raise ValueError("Unsupported modality: {}".format(modality))
    
    #create prompt
    prompt = "Classify the molecule using {} representation.".format(molecule_id, modality)
    #get LLM response
    response = get_llm_response_from_smiles(prompt, smiles, model=model)
    
    return response

def get_llm_response_from_smiles(prompt, smiles, model="gpt-4.1"):
    """Get a response from the LLM based on a SMILES string."""
    
    response = client.responses.create(
        model=model,
        input=prompt,
        context={"smiles": smiles}
    )
    
    return response.output_text
