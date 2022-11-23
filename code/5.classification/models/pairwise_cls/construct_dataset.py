import json
from pathlib import Path
import random
import numpy as np
from collections import Counter
from itertools import product
seed = 2
test_num = 10
random.seed(seed)
'''
dataset for pairwise sentence classification
'''

dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v2/Q105543609")
dir_output = dir_data / f"pairwise_cls/cross_{seed}"
dir_output.mkdir(exist_ok=True, parents=True)
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
true_test_data = []

cnt_train_entity_pair = 0
cnt_test_entity_pair = 0
cnt_dev_entity_pair = 0

# test_num = int(len(labeled_data) * 0.1)

example_ids = list(range(len(labeled_data)))
random.shuffle(example_ids) # shuffle data
print(example_ids[:10])

# all_entities = set()
# for i in example_ids:
#     obj = labeled_data[i]
#     e1, e2 = obj["entity_pair"]
#     all_entities.add(e1)
#     all_entities.add(e2)
# test_entities = set(random.sample(all_entities, test_num))
# dev_entities = set(random.sample(all_entities - test_entities, test_num))



test_entities = set()
test_entity_pairs = set()
dev_entities = set()
dev_entity_pairs = set()

for i in example_ids:
    obj = labeled_data[i]
    e1, e2 = obj["entity_pair"]
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
# dev_entities = set()
# dev_entity_pairs = set()
# for i in example_ids:
#     obj = labeled_data[i]
#     e1, e2 = obj["entity_pair"]
    
#     if e1 not in test_entities and e2 not in test_entities: # a valid dev example
#         dev_entities.add(e1)
#         # if len(dev_entities) >= len(test_entities):
#         #     break
#         dev_entities.add(e2)
#         # if len(dev_entities) >= len(test_entities):
#         #     break
#         dev_entity_pairs.add((e1, e2))
#         if len(dev_entity_pairs) >= len(test_entity_pairs):
#             break
# print("dev_entities", len(dev_entities))
# print("dev_entity_pairs", len(dev_entity_pairs))


train_entities = set()
for k,i in enumerate(example_ids):
    obj = labeled_data[i]
    # keys: "entity_pairs", "positive_labels"
    e1, e2 = obj["entity_pair"]
    texts_e1 = texts[e1]
    texts_e2 = texts[e2]
    positive_sent_pair = set()
    non_negative_sent_pair = set() # not selected as positive pair but should not be selected as negative pairs as well
    positive_data = []
    if len(texts_e1) == len(texts_e2) == 1: # delete one-sentence examples because we want to have negatives
        continue
    # positive
    counter_statement_pair = Counter()
    for j,item in enumerate(obj["positive_labels"]):
        sent_id1, sent_id2, statement_pairs = item
        strong_pos = True
        for sp in statement_pairs:
            sp = tuple(sp)
            if counter_statement_pair[sp] > 5:
                strong_pos = False
                break
            counter_statement_pair[sp] += 1
        if not strong_pos:
            non_negative_sent_pair.add((sent_id1, sent_id2))
            continue
        
        positive_sent_pair.add((sent_id1, sent_id2))

        new_obj = {
            "id": f"pos_{i}_{sent_id1}_{sent_id2}",
            "text_e1": texts_e1[sent_id1],
            "text_e2": texts_e2[sent_id2],
            "label": 1,
        }
        positive_data.append(new_obj)

    # if len(positive_sent_pair) > 100:
    #     print(i)
    # random negative
    neg_sent_pair = set()
    sent_id1_range = list(range(len(texts_e1)))
    random.shuffle(sent_id1_range)
    sent_id2_range = list(range(len(texts_e2)))
    random.shuffle(sent_id2_range)
    cnt = 0
    negative_data = []
    for sent_id1, sent_id2 in product(sent_id1_range, sent_id2_range):
        if cnt >= len(positive_sent_pair):
            break
        sent_id1 = np.random.randint(0, len(texts_e1))
        sent_id2 = np.random.randint(0, len(texts_e2))
    # for sent_id1 in sent_id1_range:
    #     if cnt == len(positive_sent_pair):
    #         break
    #     for sent_id2 in sent_id2_range:
    #         if cnt == len(positive_sent_pair):
    #             break
        if ((sent_id1, sent_id2) not in positive_sent_pair) and ((sent_id1, sent_id2) not in non_negative_sent_pair) and ((sent_id1, sent_id2) not in neg_sent_pair):
            if texts_e1[sent_id1].count(' ') < 5 or texts_e2[sent_id2].count(' ')< 5:
                continue
            neg_sent_pair.add((sent_id1, sent_id2))
            new_obj = {
                "id": f"neg_{i}_{sent_id1}_{sent_id2}",
                "text_e1": texts_e1[sent_id1],
                "text_e2": texts_e2[sent_id2],
                "label": 0,
            }
            cnt += 1
            negative_data.append(new_obj)
    # if e1 in test_entities and e2 in test_entities: # entity pairs # both e1 and e2 in test_entities
    if (e1, e2) in test_entity_pairs:
        test_data.extend(positive_data)
        test_data.extend(negative_data)
        # # generate true test (enumerate all sentence pairs)
        # for _id1 in range(len(texts_e1)):
        #     for _id2 in range(len(texts_e2)):
        #         if (_id1, _id2) in positive_sent_pair:
        #             _label = "pos"
        #         else:
        #             _label = "neg"
        #             new_obj = {
        #                 "id": f"{_label}_{i}_{_id1}_{_id2}",
        #                 "text_e1": texts_e1[_id1],
        #                 "text_e2": texts_e2[_id2],
        #                 "label": 0,  
        #             }
        #             true_test_data.append(new_obj)

        cnt_test_entity_pair += 1
    elif (e1, e2) in dev_entity_pairs:
    # elif e1 in dev_entities and e2 in dev_entities:
        dev_data.extend(positive_data)
        dev_data.extend(negative_data)
        cnt_dev_entity_pair += 1
    elif e1 not in (dev_entities | test_entities) and e2 not in (dev_entities | test_entities):
        train_data.extend(positive_data)
        train_data.extend(negative_data)
        train_entities.add(e1)
        train_entities.add(e2)
        cnt_train_entity_pair += 1
    else:
        continue
    if len(neg_sent_pair) < len(positive_sent_pair):
        continue
    if len(neg_sent_pair) != len(positive_sent_pair):
        print(len(neg_sent_pair))
        print(len(positive_sent_pair))
        print(texts_e1)
        print(texts_e2)
    assert len(neg_sent_pair) == len(positive_sent_pair)

print("train entities", len(train_entities))
assert len(test_entities) + len(dev_entities) + len(train_entities) == len(test_entities | dev_entities | train_entities)

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

