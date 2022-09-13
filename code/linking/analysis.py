from pathlib import Path
import json
import os
from collections import Counter
from nltk.tokenize import sent_tokenize

dir_data = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data/")
split_list = ["AA", "AB"]


for split in split_list:
    for batch_file in os.listdir(dir_data / split):
        with open(dir_data / split / batch_file) as f:
            for line in f:
                cnt_pages += 1
                obj = json.loads(line)
                for s in sent_tokenize(obj["text"]):
                    if len(s) > 15:
                        cnt_
print(cnt_pages, cnt_has_text)