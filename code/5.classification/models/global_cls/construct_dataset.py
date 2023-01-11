import json
from pathlib import Path
import random
import numpy as np
from collections import Counter
from itertools import product
from collections import Counter
import random
random.seed(7)

'''
{
    "id": "Q5_s1_s2",
    "text_e1": sentences_e1,
    "text_e2": sentences_e2,
    "labels": ]

}

'''

dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/news_v1")
dir_output = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/global_sentence_extraction/news_v1")
dir_output.mkdir(exist_ok=True, parents=True)

all_train_data = []
all_dev_data = []

for x in dir_data.glob("Q*"):
    etype = x.stem
    dir_etype = dir_data / etype
    # load texts
    path_texts = dir_etype / "texts.json"
    texts = {}
    with open(path_texts) as f:
        for line in f:
            obj = json.loads(line)
            texts[obj["qid"]] = obj["sentences"]

    etype_data = []
    etype_entities = set()
    entity_pairs = set()
    with open(dir_etype / "labels.json") as f:
        for line in f:
            obj = json.loads(line)
            e1, e2 = obj["entity_pair"]
            statement_cnt = Counter()
            # set two sliding windows
            texts_e1 = texts[e1]
            texts_e2 = texts[e2]
            real_min_sent_e1 = obj["positive_labels"][0][0]
            real_min_sent_e2 = obj["positive_labels"][0][1]
            real_max_sent_e1 = obj["positive_labels"][-1][0]
            real_min_sent_e2 = obj["positive_labels"][0][1]
            for i in range(int(len(texts_e1) / 10) + 1):
                min_sent_e1 = i * 10
                max_sent_e1 = min(min_sent_e1 + 10, len(texts_e1))
                if max_sent_e1 < real_min_sent_e1 or min_sent_e1 > real_max_sent_e1:
                    continue
                for j in range(int(len(texts_e2) / 10) + 1):
                    min_sent_e2 = j * 10
                    max_sent_e2 = min(min_sent_e2 + 10, len(texts_e2))
                    if max_sent_e2 < real_min_sent_e2 or min_sent_e2 > real_max_sent_e2:
                        continue
                    positive_labels = []

                    
                    for sent_id1, sent_id2, statement_info in obj["positive_labels"]:
                        # if outside the text window, skip
                        if sent_id1 < min_sent_e1 or sent_id1 >= max_sent_e1:
                            continue
                        if sent_id2 < min_sent_e2 or sent_id2 >= max_sent_e2:
                            continue      
                        news_freq = 0  # for this sentence pair           
                        for s_info in statement_info:
                            if statement_cnt[tuple(s_info)] >= 3:
                                continue
                            news_freq += s_info[2]
                            statement_cnt[tuple(s_info)] += 1
                        # sent_id1, sent_id2, news_frequency
                        if news_freq > 0:
                            # print(min_sent_e1, max_sent_e1, min_sent_e2, max_sent_e2, (sent_id1 - min_sent_e1, sent_id2 - min_sent_e2, news_freq))
                            positive_labels.append((sent_id1 - min_sent_e1, sent_id2 - min_sent_e2, news_freq))

                    if len(positive_labels) != 0:

                        text_e1_window = texts_e1[min_sent_e1: max_sent_e1]
                        text_e2_window = texts_e2[min_sent_e2: max_sent_e2]
                        entity_pairs.add((e1, e2))
                        etype_entities.add(e1)
                        etype_entities.add(e2)

                        new_obj = {
                            "id": f"{etype}_{e1}_{e2}_{i}_{j}",
                            "text_e1": text_e1_window, # list of sentences
                            "text_e2": text_e2_window,
                            "positive_labels": positive_labels
                        }
                        etype_data.append(new_obj)
            

    dev_entity_pairs = set()
    dev_entities = set()
    train_entity_pairs = set([p for p in entity_pairs])
    train_entities = set()
    while len(dev_entity_pairs) < 0.04 * len(train_entity_pairs):
        sample = random.sample(train_entity_pairs, 1)[0]
        _dev_entity_pairs = set()
        # add dev
        _dev_entity_pairs.add(sample)
        new_dev_entities = (dev_entities | set([sample[0], sample[1]]))
        # remove from train
        _tmp = set()
        _tmp.add(sample)
        # re assign train pairs
        for e1, e2 in train_entity_pairs:
            if e1 in new_dev_entities or e2 in new_dev_entities: # remove from train
                _tmp.add((e1, e2))
            if e1 in new_dev_entities and e2 in new_dev_entities: # add to dev
                _dev_entity_pairs.add((e1, e2))
                new_dev_entities.add(e1)
                new_dev_entities.add(e2)

        
        # if making train too small, do not proceed
        if len(train_entity_pairs - _tmp) * 0.04 < len(dev_entity_pairs | _dev_entity_pairs):
            break
        train_entity_pairs -= _tmp
        dev_entity_pairs |= _dev_entity_pairs 
        dev_entities = new_dev_entities
        
    train_entities = set()
    for e1, e2 in train_entity_pairs:
        train_entities.add(e1)
        train_entities.add(e2)
    dev_entities = set()
    for e1, e2 in dev_entity_pairs:
        dev_entities.add(e1)
        dev_entities.add(e2)
    assert len(train_entities & dev_entities) == 0
    assert len(train_entity_pairs & dev_entity_pairs) == 0

    dev_data = []
    train_data = []
    for data in etype_data:
        _, e1, e2, _, _ = data["id"].split("_")
        if (e1, e2) in dev_entity_pairs:
        # if e1 in dev_entities and e2 in dev_entities:
            dev_data.append(data)
        if (e1, e2) in train_entity_pairs:
            train_data.append(data)
    print(etype, len(dev_data), len(train_data))
    all_dev_data.extend(dev_data)
    all_train_data.extend(train_data)

# write to file
print("all train, dev", len(all_train_data), len(all_dev_data))
with open(dir_output / "data_train.json", 'w') as f:
    for d in all_train_data:
        f.write(json.dumps(d) + '\n')
with open(dir_output / "data_dev.json", 'w') as f:
    for d in all_dev_data:
        f.write(json.dumps(d) + '\n')       