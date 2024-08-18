from transformers import Trainer, TrainingArguments,EvalPrediction
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
import argparse
import pandas as pd
from torch.utils.data import Dataset
from utilities.minbui import load_offline_model
from utilities import inference
from utilities.training import clean_and_decode, compute_bleu

# Dataset
INPUT_LENGTH = 512
LABEL_LENGTH = 50

SEP_TOKEN = "[/INST]"

# LoRA
LORA_R = 8
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1


class CustomDataset(Dataset):
    def __init__(self, prompts, outputs):
        self.prompts = prompts
        self.outputs = outputs

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        # We ignore input text, except where the labels begin
        ignore_input_text = torch.tensor([-100] * (INPUT_LENGTH + LABEL_LENGTH - len(self.outputs[index]['input_ids'][0])))
        label_ids = torch.cat((ignore_input_text, self.outputs[index]["input_ids"][0]), dim=0)
        attention_mask = self.prompts[index]['attention_mask']
        input_ids = self.prompts[index]['input_ids']


        return {'input_ids': input_ids[0], 'attention_mask': attention_mask[0], 'labels': label_ids}
    
def tokenize(prompt, tokenizer, return_tensors='pt', 
             cutoff_len=1500, padding=True, add_special_tokens=True):
    if padding:
        return tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            max_length=cutoff_len ,
            padding='max_length', 
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        )
    else:
        return tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            max_length=cutoff_len ,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        ) 

def load_data(path: str, delimiter = '\t', header = 0, shuffle = True):
    df = pd.read_csv(path, delimiter=delimiter, header=header)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    return df


def get_data(frame:pd.DataFrame):
    prompts = []
    prefix = 'Replace toxic language in the following comment with non-toxic language. Provide only one answer. Do not give explanations:\n'
    for _, row in frame.iterrows():

        user_prompt = {'role': 'user','content' : prefix + row['offensive-text'] }
        assistant_prompt = {'role': 'assistant','content' : row['style-transferred-text'] }
        prompts.append([user_prompt, assistant_prompt])

    return prompts


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This script generates the llm finetuning results on the APPDIA dataset.')
    parser.add_argument("-m", '--model_base_path', type=str, help='Folder', required=True)
    parser.add_argument("-c", '--cache_path', type=str, help='Folder', default = None)
    parser.add_argument("-n", '--model_name', type=str, help='Folder', required=True)
    parser.add_argument("-o", '--output_path', type=str, help='Folder', required=True)
    parser.add_argument("-a", '--appdia_path', type=str, help='Folder', required=True)
    parser.add_argument("-a", '--learning_rate', type=float, help='Folder', default= 5e-5)
    parser.add_argument("-a", '--gradient_acc_steps', type=str, help='Folder', default= 1)
    parser.add_argument("-e", '--epochs', type=str, help='Folder', default= 1)

    args = parser.parse_args()
    MODEL_BASE_PATH = args.model_base_path
    CACHE_PATH = args.cache_path
    MODEL_NAME = args.model_name
    OUTPUT_PATH = args.output_path
    APPDIA_PATH = args.appdia_path
    LR = args.learning_rate
    GRADIENT = args.gradient_acc_steps
    EPOCHS = args.epochs
    print("MODEL_BASE_PATH:", MODEL_BASE_PATH)
    print("CACHE_PATH:", CACHE_PATH)
    print("MODEL_NAME:", MODEL_NAME)
    print("OUTPUT_PATH:", OUTPUT_PATH)
    print("APPDIA_PATH:", APPDIA_PATH)
    print("LR:", LR)
    print("GRADIENT_ACC_STEPS:", GRADIENT)
    print("EPOCHS:",EPOCHS)



    train_df = load_data(APPDIA_PATH + 'original-annotated-data/original-train.tsv')
    #dev_df = load_data(APPDIA_PATH + 'original-annotated-data/original-dev.tsv')
    test_df = load_data(APPDIA_PATH + 'original-annotated-data/original-test.tsv', shuffle=False)


    if 'mistral' in MODEL_NAME:
        SEP_TOKEN = '[/INST]'
        model, tokenizer = load_offline_model(MODEL_NAME, MODEL_BASE_PATH, CACHE_PATH)
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules="all-linear",
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
        SEP_TOKEN = 'assistant'
        model, tokenizer = inference.load_model(MODEL_NAME, MODEL_BASE_PATH, CACHE_PATH)
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
    tokenizer.pad_token = tokenizer.eos_token
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    all_prompts = get_data(train_df)

    tokenized_prompts = []
    all_labels = []
    for prompt in all_prompts:
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt_splitted = prompt.split(SEP_TOKEN)
        output_str = prompt_splitted[-1][1:]
        input_tokenized = tokenize(prompt, tokenizer, return_tensors="pt", cutoff_len=INPUT_LENGTH+LABEL_LENGTH)
        output_tokenized = tokenize(output_str, tokenizer, return_tensors="pt", padding=False, add_special_tokens=False)
        tokenized_prompts.append(input_tokenized)
        all_labels.append(output_tokenized)


    train_data = CustomDataset(tokenized_prompts, all_labels)

    def compute_metrics(eval_pred: EvalPrediction):
        output, labels = eval_pred

        predictions, labels = clean_and_decode(output, labels, tokenizer)
        predictions = predictions.split(SEP_TOKEN)[-1]
        return compute_bleu(predictions, labels)

    trainer = Trainer(
            model=model,
            train_dataset=train_data,
            args=TrainingArguments(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=GRADIENT,
                num_train_epochs=EPOCHS,
                learning_rate=LR,
                lr_scheduler_type="cosine",
                warmup_steps=10,
                logging_steps=100,
                weight_decay=0.1,
                optim="adamw_torch",
                save_strategy="epoch",
                output_dir=OUTPUT_PATH + MODEL_NAME
            ),
            compute_metrics=compute_metrics
            
        )
    model.config.use_cache = False

    trainer.train()


    predictions = inference.LLM_inference(test_df['offensive-text'].to_numpy(), 
                                          model, 
                                          tokenizer, 
                                          task='detox',
                                          n_shot= 0, 
                                          example_path= APPDIA_PATH +'few_shot.json')
    labels = test_df['style-transferred-text'].tolist()

    predictions = [pred.split(SEP_TOKEN)[-1] for pred in predictions]
    model_name = MODEL_NAME.replace("/", "--")
    with open(OUTPUT_PATH + f'finetuning/{model_name}-finetuned--n{0}-e{EPOCHS}-outputs.txt', 'w') as file:
        for pred, label in zip(predictions, labels):
            file.write(f'{pred}\t{label} \n')
    
    bleu_score = compute_bleu(predictions, labels)

    print(f'test score: {bleu_score}')