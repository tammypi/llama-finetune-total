export CUDA_VISIBLE_DEVICES=0,1,2
torchrun --nproc_per_node=3 finetune_pp_peft_trainer.py \
    --model_path /root/autodl-tmp/models \
    --Train_dataset_path /root/autodl-tmp/UMLSE_TOKENIZED\
    --bf16 True \
    --output_dir /root/autodl-tmp/Fine_Tuning_Results/PMCandMedMCQA_Lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp_config /root/autodl-tmp/fsdpconfig\
    --tf32 True