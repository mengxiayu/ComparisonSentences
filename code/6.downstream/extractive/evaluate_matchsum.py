import json
references = []
with open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/downloaded/multi_doc_summ/matchsum_multi-news/test_multinews_brief.json") as f:
    for line in f:
        references.append(json.loads(line)["label"])
predictions = []
with open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/multinews_extractive/bert1213/predictions.txt") as f:
    next(f)
    for line in f:
        arr = line.strip().split('\t')
        idx = int(arr[0])
        preds = arr[1].split()
        predictions.append([int(preds[x]) for x in range(len(references[idx]))])

oracle = []
extractive_predictions = []
gold_summaries = []
with open("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/downloaded/multi_doc_summ/matchsum_multi-news/test_multinews_brief.json") as f:
    for j,line in enumerate(f):
        obj = json.loads(line)
        labels = obj["label"]
        sentences = obj["text"]
        _oracle = ""
        for i,x in enumerate(labels):
            if x == 1:
                _oracle += ' ' + sentences[i]
        oracle.append(_oracle)

        _ext = ""
        for i,x in enumerate(predictions[j]):
            if x == 1:
                _ext += ' ' + sentences[i]
        extractive_predictions.append(_ext)
                
        gold = ' '.join(obj["summary"])
        gold_summaries.append(gold)
        
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge1_all = []
rouge2_all = []
rougeL_all = []
references = []
cnt_empty = 0
for idx, gold_summaries in enumerate(gold_summaries):

    scores = scorer.score(extractive_predictions[idx] , gold)
    rouge1_all.append(scores["rouge1"][2])
    rouge2_all.append(scores["rouge2"][2])
    rougeL_all.append(scores["rougeL"][2])
print(sum(rouge1_all)/len(rouge1_all), sum(rouge2_all)/len(rouge2_all), sum(rougeL_all)/len(rougeL_all),)


references = [r for ref in references for r in ref]
predictions = [p for pred in predictions for p in pred]
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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
print(f"{p:.4f}\t{r:.4f}\t{f1:.4f}\t{acc:.4f}")
