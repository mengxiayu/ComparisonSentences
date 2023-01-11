#!/bin/bash
#$ -M myu2@nd.edu
#$ -m abe
#$ -pe smp 1
#$ -N MatchEV23

fsync $SGE_STDOUT_PATH &

/afs/crc.nd.edu/user/m/myu2/anaconda2/envs/bert/bin/python3.7 news_matching_ev.py
