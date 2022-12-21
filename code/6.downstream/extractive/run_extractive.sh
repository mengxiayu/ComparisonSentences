#!/bin/bash
#$ -M myu2@nd.edu
#$ -m abe
#$ -q gpu@qa-2080ti-007
#$ -pe smp 1
#$ -l gpu=1
#$ -N BERT


output_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/multinews_extractive/bert1219
data_dir=/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/downloaded/multi_doc_summ/matchsum_multi-news

/afs/crc.nd.edu/user/m/myu2/anaconda2/envs/bert/bin/python3.7 run_extractive.py \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_cache \
  --train_file ${data_dir}/train_multinews_brief_new.json \
  --validation_file ${data_dir}/val_multinews_brief.json \
  --test_file ${data_dir}/test_multinews_brief.json \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 512 \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --save_total_limit 2 \
  --preprocessing_num_workers 10 \
  --load_best_model_at_end \
  --logging_strategy epoch \
  --metric_for_best_model accuracy \
  --num_extractive 9 \


