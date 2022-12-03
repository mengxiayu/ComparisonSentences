#!/bin/bash
#$ -M myu2@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-006
#$ -pe smp 1
#$ -l gpu=1
#$ -N TEST-Pred


output_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/multi-news/test_pred_1202
data_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/downloaded/multi_doc_summ/multi-news_pairwise

CUDA_VISIBLE_DEVICES=3 /afs/crc.nd.edu/user/m/myu2/anaconda2/envs/bert/bin/python3.7 /afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/code/6.downstream/run_pairwise_cross.py \
  --model_name_or_path /afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pairwise_cls/cross_enc_combined_cross0_1120/checkpoint-5802 \
  --do_predict \
  --overwrite_cache \
  --test_file ${data_dir}/data_test.json \
  --per_device_eval_batch_size 1024 \
  --learning_rate 3e-5 \
  --max_seq_length 64 \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --preprocessing_num_workers 10 \
  # --max_predict_samples 300 \
  # --max_eval_samples 300 \
  
  # --evaluation_strategy epoch \
