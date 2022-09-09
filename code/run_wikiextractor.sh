#!/bin/bash
#$ -M myu2@nd.edu
#$ -m abe
#$ -pe smp 1

/afs/crc.nd.edu/user/m/myu2/anaconda2/envs/bert/bin/python3.7 -m wikiextractor.WikiExtractor -b 100M -o /afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikipedia/text \
    --json --processes 119 \
    --no-templates \
    /afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikipedia_dump/enwiki-20220101-pages-articles-multistream.xml.bz2
