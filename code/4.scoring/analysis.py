import json
from pathlib import Path
from collections import Counter
from labeling import _load_property2aliases

'''Analyze the statement matching (labeling) results'''

path_result = None
path_positive =  Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pair_scoring/Q5/all_positive_pairs.json")
path_prop_alias = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_aliases/0.tsv")
pid2alias = _load_property2aliases(path_prop_alias)
print("pid2alias loaded. size:", len(pid2alias))
    
cnt_property = Counter()
cnt_entity = Counter()

if path_result:
    positive_pairs = set()
    with open(path_result) as f:
        for line in f:
            obj = json.loads(line)
            p = obj["property"]
            cnt_property[p] += 1
            e1 = obj["entity_pair"][0]
            e2 = obj["entity_pair"][1]
            cnt_entity[e1] += 1
            cnt_entity[e2] += 1
            values_e1 = set([x[0][1] for x in obj["evidence_e1"]])
            values_e2 = set([x[0][1] for x in obj["evidence_e2"]])
            for v1 in values_e1:
                for v2 in values_e2:
                    positive_pairs.add((e1, e2, p, v1, v2))
else:
    with open(path_positive) as f:
        positive_pairs = json.load(f)
    for pair in positive_pairs:
        e1, e2, p, v1, v2 = pair
        cnt_entity[e1] += 1
        cnt_entity[e2] += 1 
        cnt_property[p] += 1

cnt_same = 0
for pair in positive_pairs:
    e1, e2, p, v1, v2 = pair
    if v1 == v2:
        cnt_same += 1

print(len(positive_pairs))
print (cnt_same)

print("cnt entity", len(cnt_entity))
# for k,v in cnt_entity.items():
#     print(v,k) 

for k,v in cnt_property.most_common():
    print(v, k, pid2alias[k][0])



