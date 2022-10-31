import json
from pathlib import Path
import os
from collections import Counter


dir_linked = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1")
criteria = ["combined"]
parts = ["AA", "AB"]


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

path_prop_alias = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_aliases/0.tsv")
pid2alias = _load_property2aliases(path_prop_alias)

criterion = criteria[0]

# dir_output = Path(dir_linked / criterion / "statistics")
# dir_output.mkdir(parents=False, exist_ok=False)

cnt_sentences = Counter()
cnt_statements = Counter()

pid2freq_val = Counter()
pid2freq_rel = Counter()

for part in parts:
    path = dir_linked / criterion / part
    for p in path.glob("wiki*"):
        with open(p) as f:
            for line in f:
                obj = json.loads(line)
                num_sent = 0
                num_entity_rels = 0
                num_entity_values = 0
                # for x in obj["linked_entity_rels"]:
                #     num_sent += 1
                #     num_entity_rels += len(x[2])
                for x in obj["linked_entity_values"]:
                    num_sent += 1
                    num_entity_values += len(x[2])
                    for statement in x[2]:
                        pid2freq_val[statement[0][1]] += 1

                for x in obj["linked_entity_rels"]:
                    num_sent += 1
                    num_entity_rels += len(x[2])
                    for statement in x[2]:
                        pid2freq_rel[statement[0][1]] += 1
with open ('/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1/statistics/combined/property_distribution_"relation"_statements', 'w') as f:
    for k,v in pid2freq_rel.most_common():
        f.write(f"{v}\t{k}\t{pid2alias[k][0]}\n")
    
with open ('/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1/statistics/combined/property_distribution_"value"_statements', 'w') as f:
    for k,v in pid2freq_val.most_common():
        f.write(f"{v}\t{k}\t{pid2alias[k][0]}\n")
