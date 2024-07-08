import torch
from transformers import AutoTokenizer, LlamaConfig
from modeling_llama import LlamaForCausalLM
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig

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
        layer.self_attn.gate.data = torch.ones_like(layer.self_attn.gate.data) - 6
        layer.self_attn.gate.requires_grad = True

    return model


def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False)
    return tokenizer.decode(output[0])

if __name__ == "__main__":
    model_name = "TinyLlama/TinyLlama_v1.1"
    custom_config = {"segment_size": 8, "delta_update": True, "use_cache": False}
    use_4bit = True
    # Set up the model
    model = setup_model(model_name, use_4bit=use_4bit, custom_config=custom_config)
    
    # Set up LoRA configuration
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=4,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=['k_proj', 'q_proj']
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, peft_config=peft_config)
    
    # Move model to GPU
    model = model.cuda()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Generate text
    prompt = "September 2007 In high school I decided I was going to study philosophy in college."
    generated_text = generate_text(model, tokenizer, prompt)
    
    print(generated_text)