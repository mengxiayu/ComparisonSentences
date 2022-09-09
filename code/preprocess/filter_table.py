# rove entities that doesn't link to wikipedia
from pathlib import Path
import os
import csv 

def exists_wikipedia_link(dir_wikipedia_link, qid):

    table_name = f"Q{qid[1]}.tsv"
    record = None
    with open(dir_wikipedia_link / table_name, 'r') as f:
        for line in f:
            if line.startswith(qid):
                record = line.strip().split('\t')
                break
    if record is None:
        return None
    return record[1] # title

def exist_wikipedia_index(index_filename, title):
    
    byte_flag = False
    data_length = start_byte = 0
    index_file = open(index_filename, 'r')
    csv_reader = csv.reader(index_file, delimiter=':')
    for line in csv_reader:
        if not byte_flag and title == line[2]:
            start_byte = int(line[0])
            byte_flag = True
        elif byte_flag and int(line[0]) != start_byte:
            data_length = int(line[0]) - start_byte
            break
    index_file.close()
    if data_length > 0:
        return True
    return False



def exist_entity_alias(dir_entity_label, qid):
    if len(qid) < 3:
        table_name = f"Q{int(qid[1]):02d}.tsv"
    else:
        table_name = f"Q{qid[1:3]}.tsv"
    with open (dir_entity_label / table_name, 'r') as f:
        for line in f:
            if line.startswith(qid):
                return True
    return False


# check Wikidata itself
def _load_property_label(path):
    pid2info = {}
    with open(path, 'r') as f:
        header = f.readline()
        assert header.strip().split('\t') == ["datatype", "label", "pid"]
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            datatype, label, pid = line.strip().split('\t')
            pid2info[pid] = (label, datatype)
    return pid2info

_cache_qid2flag = {}
def good_entity_values(dir_entity_label, pid2info, row):
    qid = row["qid"]
    # search cache first to avoid unneccesary I/O
    if qid in _cache_qid2flag:
        flag = _cache_qid2flag[qid]
    else:
        flag = exist_entity_alias(dir_entity_label, qid)
        _cache_qid2flag[qid] = flag
    if not flag:
        return False
    pid = row["property_id"]
    if pid not in pid2info:
        # print(f"Property {pid} not in index")
        return False
    return True

def good_entity_rels(dir_entity_label, pid2info, row):
    qid = row["qid"]
    # search cache first to avoid unneccesary I/O
    if qid in _cache_qid2flag:
        flag = _cache_qid2flag[qid]
    else:
        flag = exist_entity_alias(dir_entity_label, qid)
        _cache_qid2flag[qid] = flag
    if not flag:
        return False
    pid = row["property_id"]
    if pid not in pid2info:
        # print(f"Property {pid} not in index")
        return False
    return True
    
import argparse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None, help='path to input directory')
    parser.add_argument('--output_dir', type=str, default=None, help='path to output directory')
    return parser

args = get_arg_parser().parse_args()


input_dir = Path(args.input_dir)
# dir_wikilink = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/wikipedia_links_sorted")
output_dir = Path(args.output_dir)
pid2info = _load_property_label("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/property_labels/0.tsv") # for all
# index_path = Path("/afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikipedia_dump/enwiki-20220101-pages-articles-multistream-index.txt")
dir_entity_label = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed/aliases_sorted") # for all

print(input_dir)
print(output_dir)

# specify process function
if "entity_values" in args.input_dir:
    flag_function = good_entity_values
elif "entity_rels" in args.input_dir:
    flag_function = good_entity_rels
else:
    print("wrong input directory")

# start processing
for batch_file in os.listdir(input_dir):
    print(batch_file)
    if not batch_file.endswith("tsv"):
        continue
    with open(input_dir / batch_file, 'r') as f, open(output_dir / batch_file, "w") as fw:
        header = f.readline()
        assert header.strip().split('\t') == ["property_id", "qid", "value"]
        fw.write(header)
        cnt_read = 0
        cnt_write = 0
        while True:
            line = f.readline()
            cnt_read += 1
            if len(line) == 0:
                break
            arr = line.rstrip().split('\t')
            if len(arr) < 3:
                continue
            pid, qid, value = arr
            row = {"property_id": pid, "qid": qid, "value": value}
            flag = good_entity_rels(dir_entity_label, pid2info, row)

            # if qid in _cache_qid2flag:
            #     flag = _cache_qid2flag[qid]
            # else:
            #     title = exists_wikipedia_link(dir_wikilink, qid)
            #     if title is not None:
            #         flag = exist_wikipedia_index(index_path, title)
            #     _cache_qid2flag[qid] = flag
            if flag:
                fw.write(line)
                cnt_write += 1
            if cnt_read % 100000 == 0:
                print(f"read {cnt_read}, write {cnt_write}")
        print(f"{batch_file}, read {cnt_read}, write {cnt_write}")
