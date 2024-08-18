import os
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from utilities import data_augmentation
from typing import Literal


def load_model(model_name, base_path = '/lustre/project/ki-topml/mcascino/models/', cache_path = None):

    model_name = 'models--' + model_name.replace("/", "--")

    model_path = os.path.join(base_path, model_name)

    if cache_path is None:
        cache_dir = model_path
    else:
        cache_dir = cache_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir = cache_dir, padding_side = 'left', device_map = 'auto')
    
    if 't5' in model_path:
        model = T5ForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir, device_map = 'auto')
    elif 'bart' in model_path:
        model = BartForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir, device_map = 'auto')
        tokenizer = BartTokenizer.from_pretrained(model_path, cache_dir = cache_dir, padding_side = 'left', device_map = 'auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir = cache_dir, device_map = 'auto')

    return model, tokenizer



def T5_inference(data, model, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(data, padding = True, return_tensors = "pt" ).to(model.device)

    outputs = model.generate(**input_ids, max_new_tokens = 30)


    responses = tokenizer.batch_decode(outputs, skip_special_tokens = True)

    return responses


def LLM_inference(data, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, task = Literal['detox','med'] ,n_shot = 0, example_path = None, prompt_lang='de'): 
    """
    data = [A,B,C,D,E]
    """
    tokenizer.pad_token = tokenizer.eos_token
    messages = [
        data_augmentation.generate_prompt([entry], 
                                                    examples_path=example_path, 
                                                    n=n_shot, 
                                                    task=task,
                                                    lang=prompt_lang) for entry in data
    ]

    input_ids = [tokenizer.apply_chat_template(
        [message],
        padding = True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict = True
    ).to(model.device) for message in messages]

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = [model.generate(
        **input_id,
        max_new_tokens=64,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    ).squeeze() for input_id in input_ids]

    responses = tokenizer.batch_decode(outputs, skip_special_tokens = True)

    return responses
    
    
    