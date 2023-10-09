
# CUDA_VISIBLE_DEVICES=1 python run_glue_no_trainer.py --model_name_or_path /home/huyiwen/pretrained/bert-base-uncased-QQP --task_name qqp --max_length 256 --truncation False --per_device_train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 10 --output_dir /home/huyiwen/tmp/qqp/

# CUDA_VISIBLE_DEVICES=1 python run_glue_no_trainer.py --model_name_or_path lstm --task_name sst2 --max_length 80 --truncation False --per_device_train_batch_size 50 --learning_rate 2e-5 --num_train_epochs 20 --output_dir /home/huyiwen/tmp/sst2/

CUDA_VISIBLE_DEVICES=2 python run_glue_no_trainer.py --model_name_or_path /home/huyiwen/pretrained/bert-base-uncased-SST-2 --task_name sst2 --max_length 128 --truncation False --per_device_train_batch_size 50 --learning_rate 2e-5 --num_train_epochs 20 --output_dir /home/huyiwen/tmp/sst2/
