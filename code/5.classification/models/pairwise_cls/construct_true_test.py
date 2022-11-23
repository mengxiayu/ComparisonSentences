from pathlib import Path
import json
dir_output = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v2/Q5/pairwise_cls")
dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v2/Q5")
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
example_ids = []
small_test_data = []
with open (dir_output / "pairwise_test.json") as f:
    for line in f:
        obj = json.loads(line)
        if len(example_ids) >= 100:
            break
        _, example_id, _, _  = obj["id"].split('_')
        if int(example_id) not in example_ids:
            example_ids.append(int(example_id))
        small_test_data.append(obj)

print("# doc pairs", len(example_ids))
print("# sampled sentence pairs", len(small_test_data))

true_test_data = []
for i in example_ids:

    obj = labeled_data[i]
    # collect positive sentence pairs
    positive_sent_pair = set()
    for item in obj["positive_labels"]:
        sent_id1, sent_id2, statement_pairs = item
        positive_sent_pair.add((sent_id1, sent_id2))

    texts_e1 = texts[obj["entity_pair"][0]]
    texts_e2 = texts[obj["entity_pair"][1]]

    for _id1 in range(len(texts_e1)):
        for _id2 in range(len(texts_e2)):
            if (_id1, _id2) in positive_sent_pair:
                _label = "pos"
                new_obj = {
                    "id": f"{_label}_{i}_{_id1}_{_id2}",
                    "text_e1": texts_e1[_id1],
                    "text_e2": texts_e2[_id2],
                    "label": 1,  
                }
                true_test_data.append(new_obj)
            else:
                _label = "neg"
                new_obj = {
                    "id": f"{_label}_{i}_{_id1}_{_id2}",
                    "text_e1": texts_e1[_id1],
                    "text_e2": texts_e2[_id2],
                    "label": 0,  
                }
                true_test_data.append(new_obj)
print("# true test data", len(true_test_data))

with open(dir_output / "pairwise_true_test.json", 'w') as f:
    for obj in true_test_data:
        f.write(json.dumps(obj)+'\n')  

with open(dir_output / "pairwise_small_test.json", 'w') as f:
    for obj in small_test_data:
        f.write(json.dumps(obj)+'\n')  