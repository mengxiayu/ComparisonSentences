import json
from pathlib import Path
import random
import numpy as np
from collections import Counter
from itertools import product

'''
dataset for news generation
input: document (window that contains the linked sentence) A [SEP] document B
output: news sentence A [SEP] news sentence B

matched.json
{
    "entity_pair": [],
    "property": "",
    "evidence_e1": [
        [
            ["P102", "Q9630"],
            [alias],
            "news sentence"
        ]
    ]
}


'''

target_etype = "Q5398426"
dir_data = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v2/{target_etype}")

path_matched = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/statement_scoring/news/{target_etype}_matched.json")

properties_to_remove = ["P735", "P31"] # human-defined unwanted properties

def validate_pair(pair):
    e1, e2, p, v1, v2 = pair
    if p in properties_to_remove:
        return False
    if e1 == v2 and e2 == v1: # remove symmetric pair
        return False
    return True


def reorder_pair(pair):
    e1, e2, p, v1, v2 = pair
    if e1 > e2:
        return e2, e1, p, v2, v1
    return pair

# # load texts
# texts = {}
# with open(dir_data / "texts.json") as f:
#     for line in f:
#         obj = json.loads(line)
#         texts[obj["qid"]] = obj["sentences"]
# load statements
idx2statement = {}
statement2idx = {}
with open (dir_data / "statements.tsv") as f:
    for line in f:
        arr = line.strip().split('\t')
        idx = int(arr[0])
        s = eval(arr[1])
        idx2statement[idx] = s
        statement2idx[s] = idx


# load statement pairs
statementpair2freq = {}
with open (dir_data / "statement_pairs.tsv") as f:
    for line in f:
        s1, s2, freq = line.strip().split('\t')
        statementpair2freq[(int(s1), int(s2))] = int(freq)
        

# build statement pair to evidence
statementpair2evidence = {}
with open(path_matched) as f:        
    for line in f:
        obj = json.loads(line)
        p = obj["property"]
        e1, e2 = obj["entity_pair"]
        values_e1 = set([x[0][1] for x in obj["evidence_e1"]])
        values_e2 = set([x[0][1] for x in obj["evidence_e2"]])
        
        for x in obj["evidence_e1"]:
            v1 = x[0][1]
            evidence_1 = x[2]
            if evidence_1.count(' ') > 150:
                continue
            evidence_1 = evidence_1.replace("-LRB-", "(").replace("-RRB-", ")").replace("``", "\"").replace("''", "\"")
            for y in obj["evidence_e2"]:
                v2 = y[0][1]
                evidence_2 = y[2]
                if evidence_2.count(' ') > 150:
                    continue
                evidence_2 = evidence_2.replace("-LRB-", "(").replace("-RRB-", ")").replace("``", "\"").replace("''", "\"")
                if validate_pair((e1, e2, p, v1, v2)):
                    _e1, _e2, _p, _v1, _v2 = reorder_pair((e1, e2, p, v1, v2)) 
                    if (_e1, _p, _v1) not in statement2idx or (_e2, _p, _v2) not in statement2idx:
                        print((_e1, _p, _v1))
                        continue
                    s1_id = statement2idx[(_e1, _p, _v1)]
                    s2_id = statement2idx[(_e2, _p, _v2)]
                    freq = statementpair2freq[(s1_id, s2_id)]
                    if (s1_id, s2_id) not in statementpair2evidence:
                        statementpair2evidence[(s1_id, s2_id)] = []
                    statementpair2evidence[(s1_id, s2_id)].append((evidence_1, evidence_2))

fw = open(dir_data / "evidences.json", 'w')
cnt = 0
for pair in statementpair2freq:
    if pair not in statementpair2evidence:
        # print(pair, "not have evidence ??")
        continue
    evidence = statementpair2evidence[pair]
    cnt += 1
    obj = {"statement_pair": pair, "evidences": evidence}
    fw.write(json.dumps(obj) + '\n')
fw.close()
print(cnt)


        # pairs = [reorder_pair((e1, e2, p, v1, v2)) for v1 in values_e1 for v2 in values_e2 if validate_pair((e1, e2, p, v1, v2))]
        # e1, e2 will in order (e1 < e2), to avoid duplicate pairs
        






