import multiprocessing
from multiprocessing import Queue, Process
import time
import os
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
import json

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/afs/crc.nd.edu/group/dmsquare/vol1/data/WIKIPEDIA/wikidata', help='path to input directory')
    parser.add_argument('--table_name', type=str, default='entity_values', help='table name')
    parser.add_argument('--rel', type=str, default=None, help='relationship')
    parser.add_argument('--value', type=str, default=None, help='target value')
    parser.add_argument('--num_procs', type=int, default=10, help='Number of processes')
    parser.add_argument('--output', type=str, default="/afs/crc.nd.edu/group/dmsquare/vol2/myu2/ComparisonSentences/data/wikidata", help='path to output directory')
    parser.add_argument('--index', type=str, default=None, help='path to index directory')
    parser.add_argument('--entities_file', type=str, default=None, help="entity list file")
    parser.add_argument('--output_name', type=str, default="extracted", help="output file name")
    return parser

# 1. define functions for multiprocessor
def _read_dir(
        work_queue: Queue, 
        args
    ):
    path = Path(args.data) / args.table_name
    dir_list = os.listdir(path)
    print(f"searching in {len(dir_list)} files...")
    for batch_file in dir_list:
        # work_queue.put(path+batch_file)
        work_queue.put(path / batch_file)

def _read_filtered_dir(work_queue: Queue, args):
    assert args.entities_file is not None
    path = Path(args.data) / args.table_name
    with open (args.entities_file, 'r') as f:
        print("loading entity list from:", args.entities_file)
        entity_list = [line.strip() for line in f]

    if len(entity_list) < 200 and args.index is not None: # TODO remove index.
        print("loading index from:", args.index)
        # load index parts and look for target files
        prefix2eneities = {} # {"Q1": ["Q124235", "Q1637721", ...], "Q2": ...}
        for e in entity_list:
            if e[:2] not in prefix2eneities:
                prefix2eneities[e[:2]] = []
            prefix2eneities[e[:2]].append(e)
        print(f"loading index from {len(prefix2eneities)} files")
        file_list = set()
        for prefix, entities in prefix2eneities.items():
            entity2file = json.load(open(Path(args.index) / f"{prefix}.json", 'r'))
            for e in entities:
                file_list.update(entity2file[e])
        file_list = list(file_list)
    else:
        file_list = os.listdir(path)
    print(f"searching in {len(file_list)} files...")
    for batch_file in file_list:
        work_queue.put(path / batch_file)

def _write_data(output_queue: Queue, args):
    df_list = []
    while True:
        df_obj = output_queue.get()
        if df_obj is None:
            break
        df_list.append(df_obj)
    result = pd.concat(df_list)
    print("result length:", len(result))
    table_dir = Path(args.output) / args.table_name
    table_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(table_dir / f"{args.output_name}.tsv", sep='\t')

def _fetch_by_qid(work_queue: Queue, output_queue: Queue, args):
    '''
    work_quque: queue of input file paths

    '''
    with open (args.entities_file, 'r') as f:
        entity_list = [line.strip() for line in f]
    while True:
        file_path = work_queue.get()
        if file_path is None:
            break
        print("searching in:", file_path)
        df = pd.read_csv(file_path, sep='\t', header=0, on_bad_lines='warn')
        record = df.loc[df["qid"].isin(entity_list)] # query in the table
        if len(record) == 0:
            continue
        output_queue.put(record)

def _fetch_by_property(work_queue: Queue, output_queue: Queue, args):
    assert args.rel is not None
    assert args.value is not None
    while True:
        file_path = work_queue.get()
        if file_path is None:
            break
        print("searching in:", file_path)
        df = pd.read_csv(file_path, sep='\t', header=0, on_bad_lines='warn')
        record = df.loc[(df["property_id"] == args.rel) & (df["value"] == args.value)]
        if len(record) == 0:
            continue
        output_queue.put(record)


# 2. define work flow
def work_flow(args, read_func, work_func, write_func):
    '''
    read from data
    process with work_queue
    write with output_queue
    '''

    start_time = time.time()
    n_process = args.num_procs
    max_size = 100 * n_process
    output_queue = Queue(maxsize=max_size) 
    work_queue = Queue(maxsize=max_size)
    data_path = Path(args.data)
    output_path = Path(args.output)

    # put file names to queue
    read_process = Process(
        target=read_func,
        args=(work_queue, args)
    )
    read_process.start()

    write_process = Process(
        target=write_func,
        args=(output_queue, args)
    )
    write_process.start()
    
    work_processes = []
    for _ in range(max(1, n_process-2)):
        work_process = Process(
            target=work_func,
            args=(work_queue, output_queue, args)
        )
        work_process.daemon = True
        work_process.start()
        work_processes.append(work_process)
    read_process.join()
    for work_process in work_processes:
        work_queue.put(None)
    # Now join the work processes
    for work_process in work_processes:
        work_process.join()
    output_queue.put(None)
    write_process.join()
    print(f"Finished processing in {time.time() - start_time}s")


# define main functions
def fetch_by_entity(args):
    work_flow(args, _read_filtered_dir, _fetch_by_qid, _write_data)

def fetch_by_property_value(args):
    work_flow(args, _read_dir, _fetch_by_property, _write_data)

if __name__ == "__main__":


    args = get_arg_parser().parse_args()
    if args.entities_file is not None:
        fetch_by_entity(args)
    if args.rel is not None and args.value is not None:
        fetch_by_property_value(args)
