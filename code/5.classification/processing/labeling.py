import numpy as np
from pathlib import Path
import random
import json
import math
import csv
import pickle
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter


# path_positive = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pair_scoring/Q5/all_positive_pairs.json")
dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data_by_type/text_data_Q5")
dir_linked = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1/combined")
path_matched = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/statement_scoring/news_v1/Q5_matched.json")
dir_output = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v2/Q5/")
dir_output.mkdir(parents=True, exist_ok=True)
properties_to_remove = ["P735"] # human-defined unwanted properties

def reorder_pair(pair):
    e1, e2, p, v1, v2 = pair
    if e1 > e2:
        return e2, e1, p, v2, v1
    return pair

"""0. load matched data first (positive pairs)"""
# with open (path_positive) as f:
#     positive_pairs = json.load(f)
def load_positive_pairs(path):
    # load positive pairs from news matched data.
    positive_pair_freq = Counter()
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            p = obj["property"]
            if p in properties_to_remove: # NOTE remove unwanted properties here
                continue
            e1, e2 = obj["entity_pair"]
            values_e1 = set([x[0][1] for x in obj["evidence_e1"]])
            values_e2 = set([x[0][1] for x in obj["evidence_e2"]])
            positive_pair_freq.update([reorder_pair((e1, e2, p, v1, v2)) for v1 in values_e1 for v2 in values_e2]) # e1, e2 will in order (e1 < e2), to avoid duplicate pairs
    return positive_pair_freq

positive_pair_freq = load_positive_pairs(path_matched)
print("most frequent positive pair", positive_pair_freq.most_common(1))




statement2idx = {}
for pair in positive_pair_freq:
    e1, e2, p, v1, v2 = pair
    s1 = (e1, p, v1)
    s2 = (e2, p, v2)
    if s1 not in statement2idx:
        statement2idx[s1] = len(statement2idx)
    if s2 not in statement2idx:
        statement2idx[s2] = len(statement2idx)
with open (dir_output / "statements.tsv", 'w') as f:
    for s, idx in statement2idx.items():
        f.write(f"{idx}\t{s}\n")
print("# statements", len(statement2idx))

with open (dir_output / "statement_pairs.tsv", 'w') as f:
    for i, (k,v) in enumerate(positive_pair_freq.most_common()):
        e1, e2, p, v1, v2  = k
        s1_id = statement2idx[(e1, p, v1)]
        s2_id = statement2idx[(e2, p, v2)]
        f.write(f"{s1_id}\t{s2_id}\t{v}\n")
print("# positive statement pairs", len(positive_pair_freq))

# pair2idx = {}
# statement2idx = {}
# with open (dir_output / "statements.tsv") as f:
#     for line in f:
#         idx, s, freq = line.strip().split('\t')
#         s = eval(s)
#         e1, e2, p, v1, v2 = s
#         freq = int(freq)
#         idx = int(idx)
#         pair2idx[s] = idx


entity_set = set()
entitypair2statement = {} # in order
for pair in positive_pair_freq:
    e1, e2, p, v1, v2 = pair
    entity_set.add(e1)
    entity_set.add(e2)
    if (e1, e2) not in entitypair2statement:
        entitypair2statement[(e1, e2)] = set()
    entitypair2statement[(e1, e2)].add(pair)
        
print("# entity:", len(entity_set))
print("# entity pair, size", len(entitypair2statement))


"""1. load linked data"""
qid2linkdata = {}
for split in ["AA", "AB"]: 
    for batch_file in (dir_linked / split).glob("wiki*"):
        # print(batch_file.name)
        f = open(batch_file)
        for line in f:
            obj = json.loads(line)
            qid = obj["qid"]
            if qid not in entity_set: # only consider matched entities
                continue
            qid2linkdata[qid] = {}
            for s_type in ["linked_entity_rels", "linked_entity_values"]:
                for _sent in obj[s_type]:
                    sent_id = _sent[0]
                    sent_text = _sent[1]
                    if sent_id not in qid2linkdata[qid]:
                        qid2linkdata[qid][sent_id] = [sent_text, set()]
                    for _triple in _sent[2]:
                        _, pid, vid = _triple[0]
                        qid2linkdata[qid][sent_id][1].add((qid, pid, vid))

print("# linked entity: ", len(qid2linkdata))
                        
                        
"""2. load positive pairs"""
qid2sentdata = {} # record text and useful (linked and matched) statements
qid2sentences = {}
for split in ["AA", "AB"]: 
    for batch_file in (dir_data / split).glob("wiki*"):
        # print(batch_file.name)
        f = open(batch_file)
        for line in f:
            obj = json.loads(line)
            qid = obj["qid"]
            if qid not in qid2linkdata: # skip unlinked entities.
                continue
            text = obj["text"]
            sentences = sent_tokenize(text)
            qid2sentdata[qid] = {} # sentid2statementset
            qid2sentences[qid] = sentences
            for j, sentence in enumerate(sentences):
                if j in qid2linkdata[qid]: # if a sentence is associated with positive statement(s)
                    assert sentence == qid2linkdata[qid][j][0]
                    linked_statements = qid2linkdata[qid][j][1]
                    qid2sentdata[qid][j] = linked_statements



    
# save labels
result_label = []
cnt = 0
all_entities = set()
for entity_pair, statement_pairs in entitypair2statement.items(): # we only consider comparable entities that contain at least a pair on comparable statement. statement pairs are in correct order.
    e1, e2 = entity_pair
    if e1 not in qid2sentences or e2 not in qid2sentences:
        continue
    example = {
        "entity_pair": (e1, e2),
        "positive_labels": [],
    }

    for i, _se1_set in qid2sentdata[e1].items(): # pull all linked statements
        for j, _se2_set in qid2sentdata[e2].items():
            _positive = []
            _positive_set = set()
            for (e1, p1, v1) in _se1_set:
                for (e2, p2, v2) in _se2_set:
                    if p1 != p2:
                        continue
                    if (e1, e2, p1, v1, v2) in statement_pairs and (e1, e2, p1, v1, v2) not in _positive_set: # positive label
                        # positive sentence pair!
                        _positive_set.add((e1, e2, p1, v1, v2))
                        _positive.append((statement2idx[(e1, p1, v1)], statement2idx[(e2, p1, v2)], positive_pair_freq[pair]))
            if len(_positive) > 0:
                example["positive_labels"].append([i, j, _positive])
    if len(example["positive_labels"]) > 0:
        all_entities.add(e1)
        all_entities.add(e2)
    
    if len(example["positive_labels"]) > 0:
        result_label.append(example)
        cnt += 1
        if cnt % 2000 == 0:
            print(cnt)
    
"""3. construct CompSent data"""

# save senteneces
with open (dir_output / "texts.json", 'w') as f:
    for qid, sentences in qid2sentences.items():
        if qid not in all_entities:
            continue
        obj = {"qid": qid, "sentences": sentences}
        f.write(json.dumps(obj) + '\n')

with open(dir_output / "labels.json", 'w') as f:
    for obj in result_label:
        f.write(json.dumps(obj) + '\n')
print("labels written")
    


