from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
pred_file = "/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/seq_cls/debug_xnli/predictions.txt"

references = []
predictions = []
with open(pred_file) as f:
    next(f)
    for line in f:
        idx, pred = line.strip().split('\t')
        label, item_id, sent_id1, sent_id2 = idx.split('_')
        if item_id == "10":
            break
        predictions.append(int(pred))
        if label == "pos":
            references.append(1)
        elif label == "neg":
            references.append(0)
assert len(references) == len(predictions)

f1 = f1_score(references, predictions)
p = precision_score(references, predictions)
r = recall_score(references, predictions)
print(f1, p, r)