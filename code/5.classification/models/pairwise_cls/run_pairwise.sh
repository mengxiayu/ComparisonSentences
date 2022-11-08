#!/bin/bash
#$ -M myu2@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-006
#$ -pe smp 1
#$ -l gpu=1


output_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pairwise_cls/cross_enc_train100k_1108
data_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v2/Q5/pairwise_cls

CUDA_VISIBLE_DEVICES=3 /afs/crc.nd.edu/user/m/myu2/anaconda2/envs/bert/bin/python3.7 run_pairwise_cross.py \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file ${data_dir}/pairwise_train.json \
  --validation_file ${data_dir}/pairwise_dev.json \
  --test_file ${data_dir}/pairwise_test.json \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 4 \
  --max_seq_length 128 \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --save_strategy epoch \
  --save_total_limit 2 \
  --preprocessing_num_workers 10 \
  --max_train_samples 100000 \
  # --max_eval_samples 300 \
  # --max_predict_samples 300 \
