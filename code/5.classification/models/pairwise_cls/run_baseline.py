from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
import json
from pathlib import Path


def run_sentence_bert():
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


    path_test = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v2/combined/cross_0/pairwise_test.json")
    dir_output = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pairwise_cls/sbert_combined_cross_0/")
    dir_output.mkdir(parents=True, exist_ok=True)

    #Load AutoModel from huggingface model repository
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    fw = open(dir_output / "predictions.txt", 'w')
    with open (path_test) as f:
        for line in f:
            obj = json.loads(line)
            idx = obj['id']
            sentences = [obj["text_e1"], obj["text_e2"]]
            encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                # print(sentence_embeddings.shape)
                cos_sim = cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0)
            fw.write(f"{idx}\t{cos_sim}\n")
    fw.close()

import evaluate
def run_bleu():
    path_test = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/sentence_scoring/v2/combined/cross_0/pairwise_test.json")
    dir_output = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/experiments/pairwise_cls/bleu_combined_cross_0/")
    dir_output.mkdir(parents=True, exist_ok=True)
    bleu = evaluate.load("bleu")
    fw = open(dir_output / "predictions.txt", 'w')
    with open (path_test) as f:
        for line in f:
            obj = json.loads(line)
            idx = obj['id']
            sentences = [obj["text_e1"], obj["text_e2"]]
            results = bleu.compute(
                predictions=[sentences[0].lower()],
                references=[[sentences[1].lower()]],
                max_order=3
            )
            score = results['bleu']
            fw.write(f"{idx}\t{score}\n")
    fw.close()
run_bleu()

            