#!/bin/bash
#$ -M myu2@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-007
#$ -pe smp 1
#$ -l gpu=1
#$ -N TEST


output_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pairwise_cls/cross_enc_train100k_1109/checkpoint-4689/pred_true_test
data_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v2/Q5/pairwise_cls

CUDA_VISIBLE_DEVICES=2 /afs/crc.nd.edu/user/m/myu2/anaconda2/envs/bert/bin/python3.7 run_pairwise_cross.py \
  --model_name_or_path /afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pairwise_cls/cross_enc_train100k_1109/checkpoint-4689 \
  --do_predict \
  --overwrite_cache \
  --test_file ${data_dir}/pairwise_true_test.json \
  --per_device_eval_batch_size 64 \
  --learning_rate 3e-5 \
  --max_seq_length 128 \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --preprocessing_num_workers 10 \
  # --max_eval_samples 300 \
  # --max_predict_samples 300 \
  # --evaluation_strategy epoch \
