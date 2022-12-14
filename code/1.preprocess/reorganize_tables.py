from pathlib import Path
import os
import pandas as pd
alias_dir = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/entity_values")
output_dir = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/entity_values_sorted")


print(alias_dir)
print(output_dir)

def index_two_digit():
    fw_list = []
    for i in range(1,100):
        fw_list.append(open(output_dir / f"Q{i:02d}.tsv", 'w'))

    for fw in fw_list:
        fw.write("property_id\tqid\tvalue\n")
    for batch_file in os.listdir(alias_dir):
        print(batch_file)
        if not batch_file.endswith("tsv"):
            continue
        with open(alias_dir / batch_file, 'r') as f:
            header = f.readline()
            # assert header.strip().split('\t') == ["qid", "alias"]
            assert header.strip().split('\t') == ["property_id", "qid", "value"]
            while True:
                arr = f.readline()
                if len(arr) == 0:
                    break
                qid = arr.strip().split('\t')[1]
                assert qid[0] == "Q"
                if len(qid) < 3:
                    fw_list[int(qid[1])-1].write(arr)
                else:
                    fw_list[int(qid[1:3])-1].write(arr)

    for fw in fw_list:
        fw.close()  

def index_one_digit():
    fw_list = []
    for i in range(1,10):
        fw_list.append(open(output_dir / f"Q{i}.tsv", 'w'))

    for batch_file in os.listdir(alias_dir):
        print(batch_file)
        if not batch_file.endswith("tsv"):
            continue
        with open(alias_dir / batch_file, 'r') as f:
            header = f.readline()
            assert header.strip().split('\t') == ["qid", "wiki_title"]
            while True:
                arr = f.readline()
                if len(arr) == 0:
                    break
                qid = arr.strip().split('\t')[0]
                assert qid[0] == "Q"
                
                fw_list[int(arr[1])-1].write(arr)

    for fw in fw_list:
        fw.close()  
    
# index_one_digit()
index_two_digit()