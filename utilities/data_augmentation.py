
import torch, json
from typing import Literal

def translate(input, model, tokenizer, source, target):
    
    input_ids = tokenizer(f"translate {source} to {target}: {input}", return_tensors = "pt" ).input_ids.to(model.device)

    output = model.generate(input_ids, max_new_tokens = 30)

    return tokenizer.decode(output[0], skip_special_tokens=True)

def back_translate(input, model, tokenizer, intermediate):

    forward = translate(input, model, tokenizer, "German", intermediate)

    backward = translate(forward, model, tokenizer, intermediate, "German")

    return backward


def paraphrase(input, model, tokenizer):

    input_ids = tokenizer(f"paraphrase: {input}", return_tensors = "pt" ).input_ids.to(model.device)

    output = model.generate(input_ids, max_new_tokens = 30)

    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_prompt(answers, examples_path = None, n=0, task = Literal['detox', 'med'], lang = 'de'):
    """
    SUBJECT TO CHANGE!
    Assumed data format for text and examples: json
    {context, question, [incorr_answers], [corr_answers]}
    """
    # if lang == 'de':
    #     prefix = f'Die Antwortmöglichkeiten eines Medizintests werden in stilistisch einheitliche, medizinische Fachsprache übertragen: \n'
    # else:
    #     raise NotImplementedError

    if task == 'detox':
        if n!=0:
            prefix = 'The following examples replace toxic language with non-toxic language.\n'
            suffix = 'Provide only one answer. No alternative phrasings. Do not give explanations. Paraphrase like in the examples:\n'
        else:
            prefix = ''
            suffix = 'Replace toxic language in the following comment with non-toxic language. Provide only one answer. Do not give explanations:\n'
    elif task == 'med':
        prefix = f'Die Antwortmöglichkeiten eines Medizintests werden in stilistisch einheitliche, medizinische Fachsprache übertragen:\n'
        suffix = 'Übertrage die folgenden Antwortmöglichkeiten in stilistisch einheitliche, medizinische Fachsprache:\n'
    else:
        raise NotImplementedError
    
    def generate_prompt_example(incorr_answers, label = None):

        if label is None:
            corr_answers = ['']*len(incorr_answers)
        else:
            corr_answers = label

        prompt = ''.join([inc + ' // ' + cor + '\n' for inc, cor in zip(incorr_answers, corr_answers)])
        
        return prompt


    def collect_examples(n = 0, data_path = 'data/data_examples.json'):
        if n<=0:
            return ''

        with open(data_path) as f:
            examples_raw = json.load(f)

            max_examples = n if n < len(examples_raw) else len(examples_raw)
            example_list = [f'Beispiel {x+1}:\n' + generate_prompt_example([examples_raw[x]['incorr_answers']], [examples_raw[x]['corr_answers']]) for x in range(max_examples)]

        examples = '\n\n'.join(example_list) + '\n'

        examples = '\n\n'.join(example_list) + '\n'

        return examples
    
    

    examples = collect_examples(n = n, data_path=examples_path)
    task = suffix + generate_prompt_example(answers)
    
    prompt = {
            'role': 'user',
            'content' : prefix + examples + task
        }
    

    return prompt

