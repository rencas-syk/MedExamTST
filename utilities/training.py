from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EvalPrediction, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
import torch
from utilities.bleu_wrapper import bleu_wrapper



class PreTokenizedDataset(Dataset):
    def __init__(self, prompts, labels):
        self.prompts = prompts
        self.labels = labels

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "input_ids": self.prompts[idx]['attention_mask'][0],
            'attention_mask': self.prompts[idx]['input_ids'][0],
            'labels': self.labels[idx]['input_ids'][0]
        }



def model_trainer(model, tokenizer, train = None, dev = None):

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        #use_cpu=True
    )
    def compute_metrics(eval_pred: EvalPrediction):
        output, labels = eval_pred

        predictions, labels = clean_and_decode(output, labels, tokenizer)
        return compute_bleu(predictions, labels)



    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=data_collator,
        compute_metrics= compute_metrics
    )
    
    return trainer # trainer.train()

def clean_and_decode(preds, labels, tokenizer):
    logits = preds[0]
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    predictions = logits.argmax(-1)

    predictions = torch.where(labels != -100, predictions, torch.tensor(0))
    labels = torch.where(labels != -100, labels, torch.tensor(0))
    predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    return predictions, labels

def compute_bleu(predictions, labels):
    results = {}
    try:
        bleu_score = bleu_wrapper(predictions = predictions, references = labels)['bleu']
    except ZeroDivisionError:
        bleu_score = 0
    
    results['bleu'] = bleu_score

    return results
