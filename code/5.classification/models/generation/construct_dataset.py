
import json
from pathlib import Path
import random
import numpy as np
from collections import Counter
from itertools import product

target_etype = "Q105543609"
dir_data = Path(f"/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v2/{target_etype}")
dir_output = dir_data / f"news_generation/"
dir_output.mkdir(exist_ok=True, parents=True)
path_texts = dir_data / "texts.json"
path_labels = dir_data / "labels.json"
path_evidences = dir_data / "evidences.json"


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
        

# load wiki texts and labels
texts = {}
with open(path_texts) as f:
    for line in f:
        obj = json.loads(line)
        texts[obj["qid"]] = obj["sentences"]

statement2sent = {} # to sentence position
with open(path_labels) as f:
    for line in f:
        obj = json.loads(line)
        for sent1, sent2, statements in obj["positive_labels"]:
            for s1, s2, freq in statements:
                if (s1, s2) not in statement2sent:
                    statement2sent[(s1, s2)] = []
                statement2sent[(s1, s2)].append((sent1, sent2))
        
for k,v in statement2sent.items():
    print(k,v)
    break

print("# text", len(texts))
print("# statement data", len(statement2sent))

def pick_context(sent_id, sentences):
    if ' '.join(sentences[:sent_id]).count(' ') < 400:
        context = ""
        i = 0
        while context.count(' ') < 400 and i < len(sentences):
            context += ' ' + sentences[i]
            i += 1
        return context
    else:
        context = sentences[:0] + sentences[sent_id-10 : sent_id+10]
        return ' '.join(context)

train_data = [] # [{"text_e1", "text_e2", "label"}]
dev_data = []
test_data = []
true_test_data = []

cnt_train_data = 0
cnt_test_data = 0
cnt_dev_data = 0
train_entity_pairs = set()



labeled_data = []
with open (path_evidences) as f:
    for line in f:
        obj = json.loads(line)
        labeled_data.append(obj)
test_num = int(len(labeled_data) * 0.1)

example_ids = list(range(len(labeled_data)))
random.shuffle(example_ids) # shuffle data
print(example_ids[:10])


test_entities = set()
test_entity_pairs = set()
dev_entities = set()
dev_entity_pairs = set()

for i in example_ids:
    obj = labeled_data[i]
    s1_id, s2_id = obj["statement_pair"]
    e1 = idx2statement[s1_id][0]
    e2 = idx2statement[s2_id][0]
    # print(i)
    # if (e1 in test_entities and e2 in dev_entities) or (e2 in test_entities and e1 in dev_entities):
    #     continue # if overlaps, then get rid of this pair
    if len(test_entity_pairs) <= len(dev_entity_pairs) and e1 not in dev_entities and e2 not in dev_entities:
        test_entity_pairs.add((e1, e2))
        test_entities.add(e1)
        test_entities.add(e2)
        # print("test", i,test_entities)
    elif e1 not in test_entities and e2 not in test_entities:
        dev_entities.add(e1)
        dev_entities.add(e2)
        dev_entity_pairs.add((e1, e2))
        # print("dev", i, dev_entities)
    else:
        continue
    if len(test_entity_pairs) >= test_num and len(dev_entity_pairs) >= test_num:
        break
    # test_entities.add(e2)
    # if len(test_entities) >= test_num:
    #     break
print("test_entities", len(test_entities))
print("test_entity_pairs", len(test_entity_pairs))
print("dev_entities", len(dev_entities))
print("dev_entity_pairs", len(dev_entity_pairs))
assert len(dev_entities & test_entities) == 0


# load evidence (target)



for k,i in enumerate(example_ids):
    obj = labeled_data[i]
    s1_id, s2_id = obj["statement_pair"]
    e1 = idx2statement[s1_id][0]
    e2 = idx2statement[s2_id][0]
    assert idx2statement[s1_id][1] == idx2statement[s2_id][1]
    p = idx2statement[s1_id][1]
    sentence_positions = statement2sent[(s1_id, s2_id)]
    
    
    context_e1 = pick_context(sentence_positions[0][0], texts[e1])
    context_e2 = pick_context(sentence_positions[0][1], texts[e2])
    new_obj = {
        "id": f"{e1}_{e2}_{p}",
        "context_e1": context_e1,
        "context_e2": context_e2,
        "evidences": obj["evidences"]
    }
    if (e1, e2) in test_entity_pairs:
        test_data.append(new_obj)
    elif (e1, e2) in dev_entity_pairs:
        dev_data.append(new_obj)
    elif e1 not in (dev_entities | test_entities) and e2 not in (dev_entities | test_entities):
        train_data.append(new_obj)
        train_entity_pairs.add((e1, e2)) 
with open(dir_output / "newsgen_train.json", 'w') as f:
    for obj in train_data:
        f.write(json.dumps(obj)+'\n')

with open(dir_output / "newsgen_dev.json", 'w') as f:
    for obj in dev_data:
        f.write(json.dumps(obj)+'\n')  

with open(dir_output / "newsgen_test.json", 'w') as f:
    for obj in test_data:
        f.write(json.dumps(obj)+'\n')  
print("train entity pairs", len(train_entity_pairs))






