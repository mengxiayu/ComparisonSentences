#!/bin/bash
#$ -M myu2@nd.edu
#$ -m abe
#$ -pe smp 1
#$ -N DumpText

/afs/crc.nd.edu/user/m/myu2/anaconda2/envs/bert/bin/python3.7 dump_text_data.py
