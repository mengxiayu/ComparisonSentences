import json
with open ("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v0/Q5/texts.json") as f:
    qid2text = {}
    for line in f:
        obj = json.loads(line)
        qid2text[obj["qid"]] = obj["sentences"]

with open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v0/Q5/labels.json") as f:
    for line in f:
        obj = json.loads(line)
        e1, e2 = obj["entity_pair"]
        for label in obj["positive_labels"]:
            
            sent_e1 = qid2text[e1][label[0]]
            sent_e2 = qid2text[e2][label[1]]
        print(sent_e1, "<--->", sent_e2)



# text
import scipy.stats as stats
print("# articles", len(qid2text))
num_sentences = []
for qid, texts in qid2text.items():
    num_sentences.append(len(texts))


    