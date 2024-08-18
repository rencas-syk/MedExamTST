import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_offline_model(model_name, model_path = "/lustre/project/ki-topml/minbui/projects/models/", cache_dir="/lustre/project/ki-topml/minbui/projects/models/cache", load_in_4_bit = False):
    def get_first_folder(directory):
        # Get the list of contents in the directory
        contents = os.listdir(directory)
        # Iterate through the contents to find the first folder
        for item in contents:
            # Check if the item is a directory
            if os.path.isdir(os.path.join(directory, item)):
                return os.path.join(directory, item)  # Return the first folder found
        return None  # Return None if no folders found

    # Load Tokenizer & Model from local
    model_name = model_name.replace("/", "--")
    model_dir = model_path + 'models--{}/snapshots'.format(model_name)
    model_id = get_first_folder(model_dir)
    if model_id is None:
        raise ValueError("No Model Found in {}!".format(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                              cache_dir=cache_dir, 
                                              padding_side='left',
                                              device_map="auto")
    tokenizer.pad_token = "!" #Not EOS, will explain another time.\
    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 torch_dtype=torch.float16, 
                                                 cache_dir=cache_dir,
                                                 device_map="auto")
    #model = None
    return model, tokenizer