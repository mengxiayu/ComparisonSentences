#!/bin/bash
#$ -M myu2@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-007
#$ -pe smp 1
#$ -l gpu=1
#$ -N bart

fsync $SGE_STDOUT_PATH &

data_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/downloaded/multi_doc_summ/multinews_abstractive
output_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/multinews_abstractive/bart1214_01

/afs/crc.nd.edu/user/m/myu2/anaconda2/envs/bert/bin/python3.7 run_seq2seq.py \
  --model_name_or_path facebook/bart-base \
  --train_file ${data_dir}/train.json \
  --validation_file ${data_dir}/validation.json \
  --test_file ${data_dir}/test.json \
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
  --max_answer_length 256 \
  --generation_max_length 256 \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --save_strategy epoch \
  --save_total_limit 2 \
  --predict_with_generate \
  --preprocessing_num_workers 10 \
  # --max_train_samples 10 \
  # --max_eval_samples 10 \
  # --max_predict_samples 10 \
