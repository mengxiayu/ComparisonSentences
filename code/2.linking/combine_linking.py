import json
from pathlib import Path
import os

criteria = ["eav", "av", "ev"]
split_list = ["AA", "AB"]
dir_linked = Path("/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/linked_v1")

def load_objects(data):
    id2object = {}
    for line in data:
        obj = json.loads(line)
        id2object[obj["qid"]] = obj
    return id2object

def combine_objects(objects):
    if len(objects) == 1:
        return objects[0]
    for obj in objects[1:]:
        assert obj["qid"] == objects[0]["qid"]
    new_obj = {
        "id": objects[0]["id"],
        "title": objects[0]["title"],
        "qid": objects[0]["qid"],
        "linked_entity_rels": [],
        "linked_entity_values": []
    }
    sent_id2entity_rels = {} # id: [text, set_of_statements, statements]
    sent_id2entity_values = {}
    
    for obj in objects:
        for sent_id, text, triples in obj["linked_entity_rels"]:
            if sent_id not in sent_id2entity_rels:
                sent_id2entity_rels[sent_id] = [text, set(), []]
            for _t in triples:
                t = ' '.join(_t[0])
                if t in sent_id2entity_rels[sent_id][1]: # already has this triple:
                    continue
                sent_id2entity_rels[sent_id][1].add(t)
                sent_id2entity_rels[sent_id][2].append(_t)

        for sent_id, text, triples in obj["linked_entity_values"]:
            if sent_id not in sent_id2entity_values:
                sent_id2entity_values[sent_id] = [text, set(), []]
            for _t in triples:
                t = ' '.join(_t[0])
                if t in sent_id2entity_values[sent_id][1]: # already has this triple:
                    continue
                sent_id2entity_values[sent_id][1].add(t)
                sent_id2entity_values[sent_id][2].append(_t)
    for sent_id, item in sorted(sent_id2entity_rels.items()):
        new_obj["linked_entity_rels"].append([sent_id, item[0], item[2]])

    for sent_id, item in sorted(sent_id2entity_values.items()):
        new_obj["linked_entity_values"].append([sent_id, item[0], item[2]])

    return new_obj
            

for split in split_list:
    output_dir = dir_linked / "combined" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch_file in os.listdir(dir_linked / criteria[0] / split):
        with open(dir_linked / criteria[0] / split / batch_file) as f:
            data_eav = load_objects(f.readlines())
        with open(dir_linked / criteria[1] / split / batch_file) as f:
            data_av = load_objects(f.readlines())
        with open(dir_linked / criteria[2] / split / batch_file) as f:
            data_ev = load_objects(f.readlines())
        combined_qid_set = set()
        combined_qid_set.update(list(data_eav.keys()))
        combined_qid_set.update(list(data_av.keys()))
        combined_qid_set.update(list(data_ev.keys()))  
        with open (output_dir / batch_file, 'w') as f:
            for qid in combined_qid_set:
                obj_list = [data[qid] for data in [data_eav, data_av, data_ev] if qid in data]
                combined_obj = combine_objects(obj_list)
                f.write(json.dumps(combined_obj) + '\n')



