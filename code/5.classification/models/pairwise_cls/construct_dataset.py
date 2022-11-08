import json
from pathlib import Path
import random
random.seed(10)
'''
dataset for pairwise sentence classification
'''

dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v2/Q5")
dir_output = dir_data / "pairwise_cls"

path_texts = dir_data / "texts.json"

path_labels = dir_data / "labels.json"

labeled_data = []
with open(path_labels) as f:
    for line in f:
        labeled_data.append(json.loads(line))
texts = {}
with open(path_texts) as f:
    for line in f:
        obj = json.loads(line)
        texts[obj["qid"]] = obj["sentences"]

print("# labeled data", len(labeled_data))
print("# text", len(texts))



train_data = [] # [{"text_e1", "text_e2", "label"}]
dev_data = []
test_data = []
cnt_train_entity_pair = 0
cnt_test_entity_pair = 0
cnt_dev_entity_pair = 0

# test_num = int(len(labeled_data) * 0.1)
test_num = 300
random.shuffle(labeled_data) # shuffle data
test_entities = set()
for i,obj in enumerate(labeled_data[:test_num]):
    e1, e2 = obj["entity_pair"]
    test_entities.add(e1)
    test_entities.add(e2)
print("test_entities", len(test_entities))

dev_entities = set()
for i, obj in enumerate(labeled_data[test_num:]):
    if len(dev_entities) > len(test_entities):
        break
    e1, e2 = obj["entity_pair"]
    if e1 not in test_entities and e2 not in test_entities: # a valid dev example
        dev_entities.add(e1)
        dev_entities.add(e2)
    
print("dev_entities", len(dev_entities))

train_entities = set()
for i,obj in enumerate(labeled_data):
    # keys: "entity_pairs", "positive_labels"
    e1, e2 = obj["entity_pair"]
    texts_e1 = texts[e1]
    texts_e2 = texts[e2]
    positive_sent_pair = set()
    positive_data = []
    if len(texts_e1) == len(texts_e2) == 1: # delete one-sentence examples because we want to have negatives
        continue
    # positive
    for j,item in enumerate(obj["positive_labels"]):
        sent_id1, sent_id2, statement_pairs = item
        positive_sent_pair.add((sent_id1, sent_id2))

        new_obj = {
            "id": f"pos_{i}_{sent_id1}_{sent_id2}",
            "text_e1": texts_e1[sent_id1],
            "text_e2": texts_e2[sent_id2],
            "label": 1,
        }
        positive_data.append(new_obj)

    if i in range(test_num) or (e1 in test_entities and e2 in test_entities): # entity pairs # both e1 and e2 in test_entities
        test_data.extend(positive_data)
        cnt_test_entity_pair += 1
    elif i in range(test_num) or (e1 in dev_entities and e2 in dev_entities):
        dev_data.extend(positive_data)
        cnt_dev_entity_pair += 1
    elif e1 in (dev_entities | test_entities) or e2 in (dev_entities | test_entities):
        continue
    else:
        train_data.extend(positive_data)
        train_entities.add(e1)
        train_entities.add(e2)
        cnt_train_entity_pair += 1
    
    # random negative
    neg_sent_pair = set()
    sent_id1_range = list(range(len(texts_e1)))
    random.shuffle(sent_id1_range)
    sent_id2_range = list(range(len(texts_e2)))
    random.shuffle(sent_id2_range)
    cnt = 0
    negative_data = []
    for sent_id1 in sent_id1_range:
        if cnt == len(positive_sent_pair):
            break
        for sent_id2 in sent_id2_range:
            if cnt == len(positive_sent_pair):
                break
            if ((sent_id1, sent_id2) not in positive_sent_pair) and ((sent_id1, sent_id2) not in neg_sent_pair):
                neg_sent_pair.add((sent_id1, sent_id2))
                new_obj = {
                    "id": f"neg_{i}_{sent_id1}_{sent_id2}",
                    "text_e1": texts_e1[sent_id1],
                    "text_e2": texts_e2[sent_id2],
                    "label": 0,
                }
                cnt += 1
                negative_data.append(new_obj)
    if i in range(test_num) or (e1 in test_entities and e2 in test_entities): # entity pairs # both e1 and e2 in test_entities
        test_data.extend(negative_data)
    elif i in range(test_num) or (e1 in dev_entities and e2 in dev_entities):
        dev_data.extend(negative_data)
    elif e1 in (dev_entities | test_entities) or e2 in (dev_entities | test_entities):
        continue
    else:
        train_data.extend(negative_data)
    if len(neg_sent_pair) != len(positive_sent_pair):
        print(len(neg_sent_pair))
        print(len(positive_sent_pair))
        print(texts_e1)
        print(texts_e2)
    assert len(neg_sent_pair) == len(positive_sent_pair)

print("train entities", len(train_entities))
assert len(train_entities & test_entities) == 0 and len(dev_entities & test_entities) == 0 and len(train_entities & dev_entities) == 0


# filter train_data

print("train/dev/test entity pairs:", cnt_train_entity_pair, cnt_dev_entity_pair, cnt_test_entity_pair)
print("train/dev/test sentence pairs:", len(train_data), len(dev_data), len(test_data))



with open(dir_output / "pairwise_train.json", 'w') as f:
    for obj in train_data:
        f.write(json.dumps(obj)+'\n')

with open(dir_output / "pairwise_dev.json", 'w') as f:
    for obj in dev_data:
        f.write(json.dumps(obj)+'\n')  

with open(dir_output / "pairwise_test.json", 'w') as f:
    for obj in test_data:
        f.write(json.dumps(obj)+'\n')  

