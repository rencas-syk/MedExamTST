from utilities import inference
import pandas as pd
import argparse



def main():

    def load_data(path: str, delimiter = '\t', header = None):
        df = pd.read_csv(path, delimiter=delimiter, header=header)

        return df

    def coherence(predictions):
        
        messages = [{'role': 'user', 'content' : 'Answer only with yes or no. Do not give explanations. Is the following sentence a coherent English sentence:\n' + pred} for pred in predictions]

        tokenizer.pad_token = tokenizer.eos_token
        input_ids = [tokenizer.apply_chat_template(
            [message],
            padding = True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device) for message in messages]

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = [model.generate(
            input_id,
            max_new_tokens=64,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        ).squeeze() for input_id in input_ids]
        responses = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        return responses
    
    def semantic_equivalence(predictions, labels):
        messages = [{'role': 'user', 'content' : f'Answer only with yes or no. Do not give explanations. Is sentence A semantically equivalent to sentence B:\n Sentence A: {pred} \n Sentence B: {label}'} for pred, label in zip(predictions, labels)]

        tokenizer.pad_token = tokenizer.eos_token
        input_ids = [tokenizer.apply_chat_template(
            [message],
            padding = True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device) for message in messages]

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = [model.generate(
            input_id,
            max_new_tokens=64,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        ).squeeze() for input_id in input_ids]
        responses = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        return responses

    def style_preservation(predictions, labels):
        messages = [{'role': 'user', 'content' : f'Answer only with yes or no. Do not give explanations. Does sentence A preserve the linguistic style of sentence B:\n Sentence A: {pred} \n Sentence B: {label}'} for pred, label in zip(predictions, labels)]

        tokenizer.pad_token = tokenizer.eos_token
        input_ids = [tokenizer.apply_chat_template(
            [message],
            padding = True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device) for message in messages]

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = [model.generate(
            input_id,
            max_new_tokens=64,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        ).squeeze() for input_id in input_ids]
        responses = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        return responses



    model, tokenizer = inference.load_model(model_name = MODEL_NAME, base_path=MODEL_BASE_PATH, cache_path=CACHE_PATH)
    #model, tokenizer = minbui.load_offline_model('mistralai/Mistral-7B-Instruct-v0.2')

    test_data = load_data(APPDIA_PATH + 'original-annotated-data/original-test.tsv', header=0)
    original_comments = test_data['offensive-text']


    flan_10 = load_data(OUTPUT_PATH + 'google--flan-t5-large--10-outputs.txt')
    flan_10_coherence = coherence(flan_10[0])
    flan_10_semantics = semantic_equivalence(flan_10[0], original_comments)
    flan_10_style = style_preservation(flan_10[0], original_comments)

    t5_10 = load_data(OUTPUT_PATH + 'google--t5-v1_1-xxl--outputs.txt')
    t5_10_coherence = coherence(t5_10[0])
    t5_10_semantics = semantic_equivalence(t5_10[0], original_comments)
    t5_10_style = style_preservation(t5_10[0], original_comments)

    llama_n0 = load_data(OUTPUT_PATH + 'meta-llama--Meta-Llama-3-70B-Instruct--0-outputs.txt')
    llama_n0_coherence = coherence(llama_n0[0])
    llama_n0_semantics = semantic_equivalence(llama_n0[0], original_comments)
    llama_n0_style = style_preservation(llama_n0[0], original_comments)

    llama_n3 = load_data(OUTPUT_PATH + 'meta-llama--Meta-Llama-3-70B-Instruct--3-outputs.txt')
    llama_n3_coherence = coherence(llama_n3[0])
    llama_n3_semantics = semantic_equivalence(llama_n3[0], original_comments)
    llama_n3_style = style_preservation(llama_n3[0], original_comments)

    mixtral_n0 = load_data(OUTPUT_PATH + 'mistralai--Mixtral-8x7B-Instruct-v0.1--0-outputs.txt')
    mixtral_n0_coherence = coherence(mixtral_n0[0])
    mixtral_n0_semantics = semantic_equivalence(mixtral_n0[0], original_comments)
    mixtral_n0_style = style_preservation(mixtral_n0[0], original_comments)

    mixtral_n3 = load_data(OUTPUT_PATH + 'mistralai--Mixtral-8x7B-Instruct-v0.1--3-outputs.txt')
    mixtral_n3_coherence = coherence(mixtral_n3[0])
    mixtral_n3_semantics = semantic_equivalence(mixtral_n3[0], original_comments)
    mixtral_n3_style = style_preservation(mixtral_n3[0], original_comments)

    bart_10 = load_data(OUTPUT_PATH + 'facebook--bart-base--10-outputs.txt')
    bart_10_coherence = coherence(bart_10[0])
    bart_10_semantics = semantic_equivalence(bart_10[0], original_comments)
    bart_10_style = style_preservation(bart_10[0], original_comments)

    mistral_n0 = load_data(OUTPUT_PATH + 'mistralai--Mistral-7B-Instruct-v0.2--0-outputs.txt')
    mistral_n0_coherence = coherence(mistral_n0[0])
    mistral_n0_semantics = semantic_equivalence(mistral_n0[0], original_comments)
    mistral_n0_style = style_preservation(mistral_n0[0], original_comments)


    mistral_n3 = load_data(OUTPUT_PATH + 'mistralai--Mistral-7B-Instruct-v0.2--3-outputs.txt')
    mistral_n3_coherence = coherence(mistral_n3[0])
    mistral_n3_semantics = semantic_equivalence(mistral_n3[0], original_comments)
    mistral_n3_style = style_preservation(mistral_n3[0], original_comments)

    data = { 
                         'flan_10_coherence':flan_10_coherence,
                         'flan_10_semantics':flan_10_semantics, 
                         'flan_10_style':flan_10_style, 
                         
                         't5_10_coherence':t5_10_coherence,
                         't5_10_semantics':t5_10_semantics, 
                         't5_10_style':t5_10_style, 
                         
                         'llama_n0_coherence':llama_n0_coherence,
                         'llama_n0_semantics':llama_n0_semantics, 
                         'llama_n0_style':llama_n0_style, 
                         
                         'llama_n3_coherence':llama_n3_coherence,
                         'llama_n3_semantics':llama_n3_semantics, 
                         'llama_n3_style':llama_n3_style, 
                         
                         'mixtral_n0_coherence':mixtral_n0_coherence,
                         'mixtral_n0_semantics':mixtral_n0_semantics, 
                         'mixtral_n0_style':mixtral_n0_style, 
                         
                         'mixtral_n3_coherence':mixtral_n3_coherence,
                         'mixtral_n3_semantics':mixtral_n3_semantics,
                         'mixtral_n3_style':mixtral_n3_style,
                         
                         'bart_10_coherence':bart_10_coherence,
                         'bart_10_semantics':bart_10_semantics,
                         'bart_10_style':bart_10_style,
                         
                         'mistral_n0_coherence':mistral_n0_coherence,
                         'mistral_n0_semantics':mistral_n0_semantics,
                         'mistral_n0_style':mistral_n0_style,
                         
                         'mixtral_n3_coherence':mistral_n3_coherence,
                         'mistral_n3_semantics':mistral_n3_semantics,
                         'mistral_n3_style':mistral_n3_style
                         }
    for key, val in data.items():
        df = pd.DataFrame({key: val})
        df.to_csv(OUTPUT_PATH + f'/LLMJudge/{key}.tsv', sep='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script generates all LLM-as-a-Judge scores on the outputs provided by appdia_script.py')
    parser.add_argument("-m", '--model_base_path', type=str, help='Folder', required=True)
    parser.add_argument("-c", '--cache_path', type=str, help='Folder', default = None)
    parser.add_argument("-n", '--model_name', type=str, help='Folder', required=True)
    parser.add_argument("-o", '--output_path', type=str, help='Folder', required=True)
    parser.add_argument("-a", '--appdia_path', type=str, help='Folder', required=True)
    args = parser.parse_args()
    MODEL_BASE_PATH = args.model_base_path
    CACHE_PATH = args.cache_path
    MODEL_NAME = args.model_name
    OUTPUT_PATH = args.output_path
    APPDIA_PATH = args.appdia_path
    print("MODEL_BASE_PATH:", MODEL_BASE_PATH)
    print("CACHE_PATH:", CACHE_PATH)
    print("MODEL_NAME:", MODEL_NAME)
    print("OUTPUT_PATH:", OUTPUT_PATH)
    print("APPDIA_PATH:", APPDIA_PATH)

    main()