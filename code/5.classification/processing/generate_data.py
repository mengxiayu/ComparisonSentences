

import json
from pathlib import Path

dir_output = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v0/Q5/dataset")
path_label = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v0/Q5/labels.json")
path_texts = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v0/Q5/texts.json")

from collections import Counter
cnt_sents = Counter()
qid2texts = {}
with open(path_texts) as f:
    for line in f:
        obj = json.loads(line)
        # qid2texts[obj["qid"]] = obj["sentences"]
        cnt_sents[len(obj["sentences"])] += 1
for k,v in cnt_sents.most_common(50):
    print(k,v)


def generate_data():

    def process_text(sentences):
        return " ".join(sentences)

    qid2texts = {}
    with open(path_texts) as f:
        for line in f:
            obj = json.loads(line)
            qid2texts[obj["qid"]] = obj["sentences"]

    results = []
    cnt = Counter() # count number of answers

    with open(path_label) as f:
        
        for line in f:
            new_obj = {
                "id": "",
                "text_pair": [],
                "answers": [],
            }
            obj = json.loads(line)
            new_obj["id"] = "_".join(obj["entity_pair"])
            sentences_e1 = qid2texts[obj["entity_pair"][0]]
            sentences_e2 = qid2texts[obj["entity_pair"][1]]
            new_obj["text_pair"] = [process_text(sentences_e1), process_text(sentences_e2)]
            for s1, s2, statements in obj["positive_labels"]:
                new_obj["answers"].append([sentences_e1[s1], sentences_e2[s2]])
            cnt[len(obj["positive_labels"])] += 1
            results.append(new_obj)

    print("total size", len(results))
    import random
    random.seed(1)
    random.shuffle(results)
    num_val = int(len(results) * 0.1)

    # train
    with open(dir_output / "train_v0_Q5.json", 'w') as f:
        for obj in results[:-num_val*2]:
            f.write(json.dumps(obj) + '\n')

    with open(dir_output / "dev_v0_Q5.json", 'w') as f:
        for obj in results[-num_val*2:-num_val]:
            f.write(json.dumps(obj) + '\n')

    with open(dir_output / "test_v0_Q5.json", 'w') as f:
        for obj in results[-num_val:]:
            f.write(json.dumps(obj) + '\n')        

from collections import Counter
def analyze_data():

    cnt = Counter()
    cnt_input_len = Counter()
    cnt_output_len = Counter()
    with open (dir_output / "train_v0_Q5.json") as f:
        for line in f:
            obj = json.loads(line)
            input_len = obj["text_pair"][0].count(' ') +  obj["text_pair"][1].count(' ')
            output_len = sum([sent.count(' ')  for ans in obj["answers"] for sent in ans ])
            cnt_output_len[output_len] += 1
            cnt_input_len[input_len] += 1
            cnt[len(obj["answers"])] += 1
    # print("input len")
    # for k,v in cnt_input_len.most_common(50):
    #     print(k, v)
    
    print("output len")
    for k,v in cnt_output_len.most_common(50):
        print(k,v)

    print("num of answer")
    for k,v in cnt.most_common(20):
        print(k,v)

# analyze_data()