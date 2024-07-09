import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments ,pipeline
from trl import SFTTrainer
import os

model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_model="tinyllama-codewello"
dataset="fka/awesome-chatgpt-prompts"



def formatted_train(input,response)->str:
    return f"<|user|>\n{input}</s>\n<|assistant|>\n{response}</s>"


def prepare_train_data(data_id):
    data = load_dataset(data_id, split="train")
    data_df = data.to_pandas()
    data_df["text"] = data_df[["act", "prompt"]].apply(lambda x: "<|user|>\n" + x["act"] + " </s>\n<|assistant|>\n" + x["prompt"] + "</s>\n", axis=1)
    data = Dataset.from_pandas(data_df)
    return data

data = prepare_train_data(dataset)



print(data[0])

def get_model_and_tokenizer(mode_id):

    tokenizer = AutoTokenizer.from_pretrained(mode_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        mode_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_id)

peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )


training_arguments = TrainingArguments(
        output_dir=output_model,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=2,
        max_steps=250,
        fp16=True,
        # push_to_hub=True
    )

trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=1024
    )


import torch
torch.cuda.empty_cache()

trainer.train()