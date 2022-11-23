from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
pred_file = "/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pairwise_cls/cross_enc_combined_cross2_1120/predictions.txt"

references = []
predictions = []
with open(pred_file) as f:
    next(f)
    for line in f:
        idx, pred = line.strip().split('\t')
        label, item_id, sent_id1, sent_id2 = idx.split('_')
        predictions.append(int(pred))
        if label == "pos":
            references.append(1)
        elif label == "neg":
            references.append(0)
assert len(references) == len(predictions)

f1 = f1_score(references, predictions)
p = precision_score(references, predictions)
r = recall_score(references, predictions)
# acc = recall_score(references, predictions)
cnt = 0
for i in range(len(references)):
    if references[i] == predictions[i] :
        cnt += 1
acc = accuracy_score(references, predictions)
# acc = cnt / len(references)
print(f"Overall\t{p:.4f}\t{r:.4f}\t{f1:.4f}\t{acc:.4f}")


references = []
predictions = []
with open(pred_file) as f:
    next(f)
    for i,line in enumerate(f):
        idx, pred = line.strip().split('\t')
        label, item_id, sent_id1, sent_id2 = idx.split('_')
        predictions.append(int(pred))
        if label == "pos":
            references.append(1)
        elif label == "neg":
            references.append(0)
        if i == 6838:
            break
assert len(references) == len(predictions)

f1 = f1_score(references, predictions)
p = precision_score(references, predictions)
r = recall_score(references, predictions)
# acc = recall_score(references, predictions)
cnt = 0
for i in range(len(references)):
    if references[i] == predictions[i] :
        cnt += 1
acc = accuracy_score(references, predictions)
# acc = cnt / len(references)
# print("Q5:", f1, p, r, acc)
print(f"Q5\t{p:.4f}\t{r:.4f}\t{f1:.4f}\t{acc:.4f}")

references = []
predictions = []
with open(pred_file) as f:
    next(f)
    for i,line in enumerate(f):
        if i <= 6838:
            continue
        idx, pred = line.strip().split('\t')
        label, item_id, sent_id1, sent_id2 = idx.split('_')
        predictions.append(int(pred))
        if label == "pos":
            references.append(1)
        elif label == "neg":
            references.append(0)

assert len(references) == len(predictions)

f1 = f1_score(references, predictions)
p = precision_score(references, predictions)
r = recall_score(references, predictions)
# acc = recall_score(references, predictions)
cnt = 0
for i in range(len(references)):
    if references[i] == predictions[i] :
        cnt += 1
acc = accuracy_score(references, predictions)
# acc = cnt / len(references)
# print("Others:", f1, p, r, acc)
print(f"Others\t{p:.4f}\t{r:.4f}\t{f1:.4f}\t{acc:.4f}")
