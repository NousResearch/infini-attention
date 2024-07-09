import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments ,pipeline
from trl import SFTTrainer
import os

import torch
from transformers import AutoTokenizer, LlamaConfig
from modeling_llama import LlamaForCausalLM
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig

model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_model="tinyllama-codewello"
dataset="fka/awesome-chatgpt-prompts"


def setup_model(model_name, use_4bit=False, custom_config=None):
    # Set up quantization config if 4-bit quantization is requested
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model configuration
    config = LlamaConfig.from_pretrained(model_name, attn_implementation="eager")
    
    # Update config with custom values if provided
    if custom_config:
        config.__dict__.update(custom_config)

    # Load the model
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=quantization_config
    )

    # Modify self-attention gates
    for layer in model.model.layers:
        layer.self_attn.gate.data = torch.ones_like(layer.self_attn.gate.data) -5
        layer.self_attn.gate.requires_grad = True

    return model


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

_, tokenizer = get_model_and_tokenizer(model_id)
custom_config = {"segment_size": 16, "delta_update": True, "use_cache": False} 
use_4bit = False
model = setup_model(model_id, use_4bit=use_4bit, custom_config=custom_config)

peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

from peft import get_peft_model, LoraConfig

model = get_peft_model(model, peft_config=peft_config)

for layer in model.model.model.layers:
    layer.self_attn.gate.requires_grad = True

training_arguments = TrainingArguments(
        output_dir=output_model,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=1,
        num_train_epochs=2,
        max_steps=250,
        fp16=True,
        # push_to_hub=True
    )

trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        # peft_config=peft_config,
        dataset_text_field="text",
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=1024
    )


import torch
torch.cuda.empty_cache()

trainer.train()

lst = []
# Modify self-attention gates
for layer in model.model.model.layers:
    data = layer.self_attn.gate.data.detach()
    data = torch.sigmoid(data)
    print(data.reshape(-1))
    lst.append(data.reshape(-1).tolist())

print("="*10)

for item in lst:
    print(item)



import matplotlib.pyplot as plt
import numpy as np
import datetime

def save_heatmap_with_timestamp(data):
    # Generate a filename with the current date and time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'heatmap_{timestamp}.png'
    
    # Convert the list of lists to a numpy array
    data_array = np.array(data)
    
    # Plotting the heatmap
    plt.figure(figsize=(16, 16))
    plt.imshow(data_array, cmap='viridis', aspect='auto')
    
    # Adding color bar
    plt.colorbar()
    
    # Annotate each cell with the numeric value
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            plt.text(j, i, f'{data_array[i, j]:.2f}', ha='center', va='center', color='white')
    
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.title('Heatmap of List of Lists')
    
    # Save the plot to a file
    file_path = f'{filename}'
    plt.savefig(file_path)
    plt.close()
    
    return file_path



file_name = save_heatmap_with_timestamp(lst)
file_name

