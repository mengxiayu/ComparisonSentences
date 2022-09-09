from pathlib import Path
import multiprocessing
from multiprocessing import Queue, Process
import json
import argparse
import time
import os

def _load_title2qid(path):
    title2qid = {}
    for batch_file in os.listdir(path):
        with open(path / batch_file) as f:
            header = f.readline()
            assert header.strip().split('\t') == ["qid", "wiki_title"]
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                arr = line.strip().split('\t')
                if len(arr) != 2:
                    print("wrong line:", arr)
                    continue
                qid, title = arr[0], arr[1]
                title2qid[title] = qid
    return title2qid

def read_data(input_queue: Queue, input_fn, title2qid):
    # read wikipedia file and lookup qid, put in the queue
    with open(input_fn) as f:
        for line in f:
            obj = json.loads(line)
            title = obj["title"]
            qid = title2qid[title] if title in title2qid else None
            if qid is None:
                continue
            obj["qid"] = qid
            input_queue.put(obj)

def _search_statement(path, qid):

    if len(qid) < 3:
        batch_file = f"Q{int(qid[1]):02d}.tsv"
    else:
        batch_file = f"Q{qid[1:3]}.tsv"
    
    flag = False
    results = []
    with open(path / batch_file) as f:
        for line in f:
            if f"\t{qid}\t" in line:
                results.append(line.strip())
                flag = True # found the entity
            elif flag == True: # 
                break
            else:
                continue
    return results

def fetch_statement(input_queue: Queue, output_queue: Queue, entity_rels_dir, entity_values_dir):
    # receive wiki title from input queue
    # fetch statements for qid
    # put statements in the output queue
    while True:
        obj = input_queue.get()
        if obj is None:
            break
        qid = obj["qid"]
        entity_rels_statements = _search_statement(entity_rels_dir, qid)
        entity_values_statements = _search_statement(entity_values_dir, qid)
        if len(entity_rels_statements) == 0 and len(entity_values_statements) == 0:
            continue
        obj["entity_rels"] =  entity_rels_statements
        obj["entity_values"] = entity_values_statements
        output_queue.put(obj)

def write_data(output_queue: Queue, output_fn):
    cnt = 0
    with open (output_fn, 'w') as f:
        while True:
            obj = output_queue.get()
            if obj is None:
                break
            f.write(json.dumps(obj) + '\n')
            cnt += 1
            if cnt % 100 == 0:
                print(f"{cnt} lines written.")

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='', help='path to input directory')
    parser.add_argument('--input_file', type=str, default='', help='input file name')
    parser.add_argument('--wikidata_dir', type=str, default='/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata_processed', help='path to wikidata tables')
    parser.add_argument('--num_procs', type=int, default=10, help='Number of processes')
    parser.add_argument('--output_dir', type=str, default="/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikipedia/text_data/AA", help='path to output directory')
    return parser


def work_flow(args):
    start_time = time.time()
    n_process = args.num_procs
    max_size = 10 * n_process
    input_queue = Queue(maxsize=max_size)
    output_queue = Queue(maxsize=max_size)

    input_file = Path(args.input_dir) / args.input_file
    output_file = Path(args.output_dir) / args.input_file
    entity_rels_dir = Path(args.wikidata_dir) / "entity_rels_sorted"
    entity_values_dir = Path(args.wikidata_dir) / "entity_values_sorted"
    title2qid = _load_title2qid(Path(args.wikidata_dir) / "wikipedia_links")
    print("title2qid size", len(title2qid))

    read_process = Process(
        target=read_data,
        args=(input_queue, input_file, title2qid)
    )

    read_process.start()

    write_process = Process(
        target=write_data,
        args=(output_queue, output_file)
    )
    write_process.start()

    work_processes = []
    for _ in range(max(1, n_process-2)):
        work_process = Process(
            target=fetch_statement,
            args=(input_queue, output_queue, entity_rels_dir, entity_values_dir)
        )
        work_process.start()
        work_processes.append(work_process)
    
    read_process.join()
    for work_process in work_processes:
        input_queue.put(None)
    for work_process in work_processes:
        work_process.join()
    output_queue.put(None)
    write_process.join()
    print(f"Finished processing in {time.time() - start_time}s")
    
    
if __name__ == "__main__":


    args = get_arg_parser().parse_args()
    work_flow(args)