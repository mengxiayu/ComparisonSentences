from pathlib import Path
import json
from nltk.tokenize import sent_tokenize, word_tokenize
import time
import os
import argparse

'''
analyze alias:
how many and what are the duplicated alias?
'''



dir_entity_aliases = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/aliases_sorted")
path_prop_alias = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_aliases/0.tsv")


def _load_qid2aliases(dir_entity_aliases):
    qid2alias = {}
    for batch_file in os.listdir(dir_entity_aliases):
        with open(dir_entity_aliases / batch_file) as f:
            for line in f:
                qid, aliases = line.strip().split('\t')
                qid2alias[qid] = aliases
    return qid2alias

def _load_property2aliases(path):
    # from the property_aliases table
    pid2aliases = {}
    with open(path, 'r') as f:
        header = f.readline()
        assert header.strip().split('\t') == ["alias", "pid"]
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            alias, pid = line.strip().split('\t')
            if pid not in pid2aliases:
                pid2aliases[pid] = []
            pid2aliases[pid].append(alias)
    return pid2aliases

def _load_alias2pid(path):
    alias2pid = {}
    with open(path, 'r') as f:
        header = f.readline()
        assert header.strip().split('\t') == ["alias", "pid"]
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            alias, pid = line.strip().split('\t')
            if alias not in alias2pid:
                alias2pid[alias] = []
            alias2pid[alias].append(pid)
    return alias2pid
def alias2qid(dir_entity_aliases):
    alias2pid = _load_alias2pid(path_prop_alias)
    alias2qid = {}
    for batch_file in os.listdir(dir_entity_aliases):
        with open(dir_entity_aliases / batch_file) as f:
            for line in f:
                qid, aliases = line.strip().split('\t')
                aliases = aliases.split('|sep|')
                for a in set(aliases):
                    if a in alias2pid:
                        print(f"({a}, {qid}) failed. Already has: ({a}, {alias2pid[a]})")

                    


alias2qid(dir_entity_aliases)