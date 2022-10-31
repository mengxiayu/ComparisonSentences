#!/bin/bash
#$ -M myu2@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-006
#$ -pe smp 1
#$ -l gpu=1

data_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v0/Q5/dataset
output_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/seq2seq/bart_input768_output64_epoch3_lr3e-5_bsz4_2k200_1019

CUDA_VISIBLE_DEVICES=1 /afs/crc.nd.edu/user/m/myu2/anaconda2/envs/bert/bin/python3.7 run_seq2seq.py \
  --model_name_or_path facebook/bart-base \
  --train_file ${data_dir}/train_v0_Q5.json \
  --validation_file ${data_dir}/dev_v0_Q5.json \
  --question_column text_pair \
  --answer_column answers \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 768 \
  --max_answer_length 128 \
  --generation_max_length 128 \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --save_strategy epoch \
  --save_total_limit 2 \
  --predict_with_generate \
  --preprocessing_num_workers 10 \
  --max_train_samples 3000 \
  --max_eval_samples 300 \
