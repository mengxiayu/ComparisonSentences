from pathlib import Path
import os
import json

# input_folder = {old text dir}
# output_folder = {new text dir}


for batch_file in os.listdir(input_folder):
    
    with open(Path(input_folder) / batch_file) as f, open(Path(output_folder) / batch_file, 'w') as fw:
        for line in f:
            obj = json.loads(line)
            if obj["text"] == "":
                continue
            fw.write(line)