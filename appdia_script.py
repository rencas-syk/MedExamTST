from utilities.training import PreTokenizedDataset, compute_bleu, model_trainer, clean_and_decode
from utilities.minbui import load_offline_model
from utilities import inference
import pandas as pd
import argparse

def load_data(path: str, delimiter = '\t', header = 0, shuffle = True):
    df = pd.read_csv(path, delimiter=delimiter, header=header)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    return df

def tokenize_dataset(tokenizer, frame: pd.DataFrame):
    tokenized = frame.map(lambda x: tokenizer(x, return_tensors='pt'))
    prompts = tokenized['offensive-text'].tolist()
    labels = tokenized['style-transferred-text'].tolist()
    dataset = PreTokenizedDataset(prompts, labels)
    return dataset

def run_seq2seq(model_id, train, dev, test):
    model, tokenizer = inference.load_model(model_id, MODEL_BASE_PATH, CACHE_PATH)

    train_dataset = tokenize_dataset(tokenizer, train)
    dev_dataset = tokenize_dataset(tokenizer, dev)
    test_dataset = tokenize_dataset(tokenizer, test)

    trainer = model_trainer(model, tokenizer, train_dataset, dev_dataset)
    trainer.train()

    out = trainer.predict(test_dataset)
    predictions, labels = clean_and_decode(out.predictions, out.label_ids, tokenizer)

    model_name = model_id.replace("/", "--")
    with open(OUTPUT_PATH + f'{model_name}--outputs.txt', 'w') as file:
        for pred, label in zip(predictions, labels):
            file.write(f'{pred}\t{label} \n')

    bleu_score = compute_bleu(predictions, labels)
    print(f'test score: {bleu_score}')
    return


def run_llm(model_id, test, n=0):
    
    model_name = model_id.replace("/", "--")

    if 'mistral' in model_id:
        SEP_TOKEN = '[/INST]'
        model, tokenizer = load_offline_model(model_id, MODEL_BASE_PATH, CACHE_PATH)
    else:
        SEP_TOKEN = 'assistant'
        model, tokenizer = inference.load_model(model_id, MODEL_BASE_PATH, CACHE_PATH)

    if 'flan' in model_name:
        prompts = test['offensive-text'].map(lambda x: 'Replace toxic language in the following comment with non-toxic language:\n' + x)
        predictions = inference.T5_inference(prompts.tolist(), model, tokenizer)
    else:
        predictions = inference.LLM_inference(test['offensive-text'].to_numpy(), model, tokenizer, task='detox',n_shot= n, example_path= APPDIA_PATH +'few_shot.json')

    labels = [label for label in test['style-transferred-text'].tolist()]
    predictions = [pred.split(SEP_TOKEN)[-1] for pred in predictions]

    with open(OUTPUT_PATH + f'{model_name}--{n}-outputs.txt', 'w') as file:
        for pred, label in zip(predictions, labels):
            file.write(f'{pred}\t{label} \n')
    
    
    

    bleu_score = compute_bleu(predictions, labels)
    print(f'test score: {bleu_score}')
    return



def main():
    
    train_df = load_data(APPDIA_PATH + 'original-annotated-data/original-train.tsv')
    dev_df = load_data(APPDIA_PATH + 'original-annotated-data/original-dev.tsv')
    test_df = load_data(APPDIA_PATH + 'original-annotated-data/original-test.tsv', shuffle=False)

    run_seq2seq("google/t5-v1_1-xxl",  train_df, dev_df, test_df)
    run_seq2seq('google/flan-t5-large',  train_df, dev_df, test_df)
    run_seq2seq('facebook/bart-base', train_df, dev_df, test_df)


    run_llm('meta-llama/Meta-Llama-3-70B-Instruct', test=test_df, n=0)
    run_llm('mistralai/Mistral-7B-Instruct-v0.2', test=test_df, n=0)
    run_llm('mistralai/Mixtral-8x7B-Instruct-v0.1', test=test_df, n=0)


    run_llm('meta-llama/Meta-Llama-3-70B-Instruct', test=test_df, n=3)
    run_llm('mistralai/Mistral-7B-Instruct-v0.2', test=test_df, n=3)
    run_llm('mistralai/Mixtral-8x7B-Instruct-v0.1', test=test_df, n=3)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script generates the seq2seq finetuning and llm prompting results on the APPDIA dataset (not included in this repository)')
    parser.add_argument("-m", '--model_base_path', type=str, help='Folder', required=True)
    parser.add_argument("-c", '--cache_path', type=str, help='Folder', default = None)
    parser.add_argument("-o", '--output_path', type=str, help='Folder', required=True)
    parser.add_argument("-a", '--appdia_path', type=str, help='Folder', required=True)
    args = parser.parse_args()
    MODEL_BASE_PATH = args.model_base_path
    CACHE_PATH = args.cache_path
    OUTPUT_PATH = args.output_path
    APPDIA_PATH = args.appdia_path
    print("MODEL_BASE_PATH:", MODEL_BASE_PATH)
    print("CACHE_PATH:", CACHE_PATH)
    print("OUTPUT_PATH:", OUTPUT_PATH)
    print("APPDIA_PATH:", APPDIA_PATH)
    
    main()