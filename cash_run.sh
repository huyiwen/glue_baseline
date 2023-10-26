
# CUDA_VISIBLE_DEVICES=5 python run_glue_no_trainer.py --model_name_or_path mpo_bert --task_name cola --max_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/cola/ --word_embed_input 21128,1 --word_embed_output 768,1 --attention_input 768,1 --attention_output 768,1 --FFN1_input 768,1 --FFN1_output 3072,1 --FFN2_input 3072,1 --FFN2_output 768,1 --mpo_lr_factor 0.99

# CUDA_VISIBLE_DEVICES=5 python run_glue_no_trainer.py --model_name_or_path mpo_bert --task_name sst2 --max_length 128 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/sst2/ --word_embed_input 21128,1 --word_embed_output 768,1 --attention_input 768,1 --attention_output 768,1 --FFN1_input 768,1 --FFN1_output 3072,1 --FFN2_input 3072,1 --FFN2_output 768,1 --mpo_lr_factor 0.99

# CUDA_VISIBLE_DEVICES=5 python run_glue_no_trainer.py --model_name_or_path mpo_bert --task_name mrpc --max_length 256 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/mrpc/ --word_embed_input 21128,1 --word_embed_output 768,1 --attention_input 768,1 --attention_output 768,1 --FFN1_input 768,1 --FFN1_output 3072,1 --FFN2_input 3072,1 --FFN2_output 768,1 --mpo_lr_factor 0.99

# CUDA_VISIBLE_DEVICES=5 python run_glue_no_trainer.py --model_name_or_path mpo_bert --task_name stsb --max_length 256 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/stsb/ --word_embed_input 21128,1 --word_embed_output 768,1 --attention_input 768,1 --attention_output 768,1 --FFN1_input 768,1 --FFN1_output 3072,1 --FFN2_input 3072,1 --FFN2_output 768,1 --mpo_lr_factor 0.99

# CUDA_VISIBLE_DEVICES=6 python run_glue_no_trainer.py --model_name_or_path mpo_bert --task_name qqp --max_length 256 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/qpp/

# CUDA_VISIBLE_DEVICES=6 python run_glue_no_trainer.py --model_name_or_path mpo_bert --task_name mnli --max_length 256 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/mnli/

# CUDA_VISIBLE_DEVICES=6 python run_glue_no_trainer.py --model_name_or_path mpo_bert --task_name qnli --max_length 256 --per_device_train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/qnli/

# CUDA_VISIBLE_DEVICES=6 python run_glue_no_trainer.py --model_name_or_path mpo_bert --task_name rte --max_length 256 --per_device_train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/rte/

# CUDA_VISIBLE_DEVICES=6 python run_glue_no_trainer.py --model_name_or_path mpo_bert --task_name wnli --max_length 256 --per_device_train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/wnli/


# 10.22

# CUDA_VISIBLE_DEVICES=2 python run_glue_no_trainer.py --model_name_or_path mpo:/home/huyiwen/pretrained/bert-base-uncased --task_name stsb --max_length 256 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/stsb/ --word_embed_input 21128,1 --word_embed_output 768,1 --attention_input 768,1 --attention_output 768,1 --FFN1_input 768,1 --FFN1_output 3072,1 --FFN2_input 3072,1 --FFN2_output 768,1 --mpo_lr_factor 0.99 --optimizer Adam --mpo_layers ""

# CUDA_VISIBLE_DEVICES=2 python run_glue_no_trainer.py --model_name_or_path /home/huyiwen/pretrained/bert-base-uncased --task_name stsb --max_length 256 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/stsb/ --word_embed_input 21128,1 --word_embed_output 768,1 --attention_input 768,1 --attention_output 768,1 --FFN1_input 768,1 --FFN1_output 3072,1 --FFN2_input 3072,1 --FFN2_output 768,1 --mpo_lr_factor 0.99 --optimizer Adam --distil_temp 1 --teacher_model /home/huyiwen/pretrained/bert-base-uncased-STS-B

# CUDA_VISIBLE_DEVICES=5 python run_glue_no_trainer.py --model_name_or_path mpo:/home/huyiwen/pretrained/bert-tiny-uncased --task_name stsb --max_length 256 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir /home/huyiwen/tmp/stsb/ --word_embed_input 21128,1 --word_embed_output 128,1 --attention_input 128,1 --attention_output 128,1 --FFN1_input 128,1 --FFN1_output 512,1 --FFN2_input 512,1 --FFN2_output 128,1 --mpo_lr_factor 1 --optimizer Adam

# 10.23

function mpo_run_glue_no_trainer() {
    CUDA_VISIBLE_DEVICES=6 python run_glue_no_trainer.py --model_name_or_path /home/huyiwen/pretrained/bert-base-uncased --task_name $1 --max_length $9 --per_device_train_batch_size $8 --num_train_epochs 4 --learning_rate $7 --output_dir /home/huyiwen/tmp/$6/$1/ --mpo_lr_factor 0.5 --optimizer Adam --distil_temp 1 --teacher_model /home/huyiwen/pretrained/bert-base-uncased-$2 --word_embed_input $3 --word_embed_output $4 --attention_input $4 --attention_output $4 --FFN1_input $4 --FFN1_output $5 --FFN2_input $5 --FFN2_output $4 --incorrect_fallback True --mpo_layers "FFN1,FFN2,attention"
}

function run_glue_no_trainer() {
    # mpo_run_glue_no_trainer $1 $2 "30,7,1,10,15" "8,2,1,6,8" "8,4,1,6,16" "ft_distil_mpo5" $4 $4 $5
    mpo_run_glue_no_trainer $1 $2 "0" "0" "0" "ft_distil" $3 $4 $5
    # mpo_run_glue_no_trainer $1 $2 "30,7,10,15" "8,2,6,8" "8,4,6,16" "ft_distil_mpo4" $3 $4 $5
}

# run_glue_no_trainer sst2 SST-2 5e-5 64 128
# run_glue_no_trainer cola CoLA 3e-5 32 256
# run_glue_no_trainer mrpc MRPC 5e-5 32 256
# run_glue_no_trainer qqp QQP 3e-5 32 256

# run_glue_no_trainer qnli QNLI 3e-5 32 256
# run_glue_no_trainer rte RTE 3e-5 32 256
# run_glue_no_trainer wnli WNLI 3e-5 32 256
run_glue_no_trainer mnli MNLI 3e-5 32 512

# big dataset: QQP MNLI
