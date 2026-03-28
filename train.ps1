# 激活环境
conda activate lora

# 进入目录
cd D:\AI\LLaMA-Factory

# 开始训练
llamafactory-cli train `
    --stage sft `
    --do_train True `
    --model_name_or_path D:\AI\models\Qwen2.5-1.5B-Instruct `
    --preprocessing_num_workers 1 `
    --finetuning_type lora `
    --template qwen `
    --flash_attn auto `
    --dataset_dir data `
    --dataset aiemplyee `
    --cutoff_len 512 `
    --learning_rate 0.0002 `
    --num_train_epochs 3.0 `
    --max_samples 100000 `
    --per_device_train_batch_size 1 `
    --gradient_accumulation_steps 8 `
    --lr_scheduler_type cosine `
    --max_grad_norm 1.0 `
    --logging_steps 5 `
    --save_steps 100 `
    --warmup_steps 10 `
    --packing False `
    --enable_thinking False `
    --report_to swanlab `
    --swanlab_project qwen-medical `
    --output_dir saves\Qwen2.5-1.5B-Instruct\lora\train_2026-03-28 `
    --fp16 True `
    --plot_loss True `
    --trust_remote_code True `
    --ddp_timeout 180000000 `
    --include_num_input_tokens_seen True `
    --optim paged_adamw_8bit `
    --gradient_checkpointing True `
    --adapter_name_or_path saves\Qwen2.5-1.5B-Instruct\lora\train_2026-03-28-15-11-16 `
    --quantization_bit 4 `
    --quantization_method bnb `
    --double_quantization True `
    --lora_rank 8 `
    --lora_alpha 16 `
    --lora_dropout 0 `
    --lora_target all `
    --val_size 0.1 `
    --eval_strategy steps `
    --eval_steps 100 `
    --per_device_eval_batch_size 1