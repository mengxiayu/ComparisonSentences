import multiprocessing
from multiprocessing import Queue, Process
import time
import os
import pandas as pd

path = "/Users/mengxiayu/Documents/Research/WIKIPEDIA/wikidata/entity_values/"

def read_dir(work_queue):
    dir_list = os.listdir(path)
    print(len(dir_list))
    for batch_file in dir_list:
        work_queue.put(batch_file)


# define process function
def fetch_by_qid(work_queue: Queue, target):
    while True:
        batch_file = work_queue.get()
        if batch_file is None:
            break
        df = pd.read_csv(path+batch_file, sep='\t', header=0)
        record = df.loc[df["qid"] == target]
        if len(record) > 0:
            print(record)



if __name__ == "__main__":

    target = "Q22686"
    start_time = time.time()
    n_process = 4
    max_size = 100 * n_process 
    work_queue = Queue(maxsize=max_size)

    # put file names to queue
    read_process = Process(
        target=read_dir,
        args=(work_queue,)
    )
    read_process.start()
    
    work_processes = []
    for _ in range(max(1, n_process-2)):
        work_process = Process(
            target=fetch_by_qid,
            args=(work_queue, target)
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
    print(f"Finished processing in {time.time() - start_time}s")
