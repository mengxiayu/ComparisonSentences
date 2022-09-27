import json
from pathlib import Path
import os
from collections import Counter
dir_linked = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1")
criteria = ["eav", "av", "ev"]
parts = ["AA", "AB"]


criterion = criteria[0]

dir_output = Path(dir_linked / criterion / "statistics")
dir_output.mkdir(parents=False, exist_ok=False)

cnt_sentences = Counter()
cnt_statements = Counter()

for part in parts:
    path = dir_linked / criterion / part
    for p in path.glob("wiki*"):
        with open(p) as f:
            for line in f:
                obj = json.loads(line)
                num_sent = 0
                num_entity_rels = 0
                num_entity_values = 0
                for x in obj["linked_entity_rels"]:
                    num_sent += 1
                    num_entity_rels += len(x[2])
                for x in obj["linked_entity_values"]:
                    num_sent += 1
                    num_entity_values += len(x[2])
                cnt_sentences[num_sent] += 1
                cnt_statements[num_entity_rels+num_entity_values] += 1



with open(dir_output / "output.txt", 'w') as f:
    # total entity(page) number
    num_entity = sum(list(cnt_sentences.values()))
    f.write(f"Number of linked entity: {num_entity}\n")
    f.write('\n')
    # total sentence number
    num_sentence = sum([k * v for k,v in cnt_sentences.items()])
    f.write(f"Number of linked sentence: {num_sentence}\n")
    f.write("#Sentence Frequency (Top 5)\n")
    for k,v in cnt_sentences.most_common(5):
        f.write(f"{k} {v}\n")
    f.write('\n')
    # total statement number
    num_statement = sum([k * v for k,v in cnt_statements.items()])
    f.write(f"Number of linked statement: {num_statement}\n")
    f.write("#Statement Frequency (Top 5)\n")
    for k,v in cnt_statements.most_common(5):
        f.write(f"{k} {v}\n")
    

