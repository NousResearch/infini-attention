accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 2 \
--gradient-accumulate-every 4 \
--output-dir ./output/infini-attn \
--wandb Infini-Attn \
--seed 2024 \
--max-train-steps 100  \
--learning-rate 2e-5  \
--dataset PY007/slimpajama_mistral_tokenized_upsample_4096_chunk_128K \
--model output/h2o-danube2-1.8b-base  \
--seq-length 16000 \
--rope-theta 30000 \
--parallel_mode data_parallel