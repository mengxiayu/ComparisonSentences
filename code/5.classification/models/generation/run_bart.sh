#!/bin/bash
#$ -M myu2@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-006
#$ -pe smp 1
#$ -l gpu=1
#$ -N GEN13

data_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/news_generation/combined
output_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pretrain_news_gen/combined_1213

CUDA_VISIBLE_DEVICES=3 /afs/crc.nd.edu/user/m/myu2/anaconda2/envs/bert/bin/python3.7 run_seq2seq.py \
  --model_name_or_path facebook/bart-base \
  --train_file ${data_dir}/newsgen_train.json \
  --validation_file ${data_dir}/newsgen_dev.json \
  --test_file ${data_dir}/newsgen_test.json \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 1024 \
  --max_answer_length 128 \
  --generation_max_length 128 \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --save_strategy epoch \
  --save_total_limit 2 \
  --predict_with_generate \
  --preprocessing_num_workers 10 \
  # --max_train_samples 3000 \
  # --max_eval_samples 300 \
