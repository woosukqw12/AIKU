import torch
import json
import os
# run: python -u "/Users/woosuk/myGitFolder/AIKU/torchvision/models/test.py" > res
# from resnet_torch.py import *
# * 폴더 이름에 family_id가 나와있음.
# * 어떤 형식으로 json들이 올지 알고싶음
# * json directory setting. 
dir_path = "../data/sample/"
family_ids = ["F0350", "F0351", "F0352"]
AorB = ( "A(친가)", "B(외가)" ) #paternal part, maternal part 
relations = ("1.Family", "2.Individuals", "3.Age")

pathes = [ f"{dir_path}{family_ids_}/{AorB_}/{relations_}/" for family_ids_ in family_ids for AorB_ in AorB for relations_ in relations ]
for i in pathes: print(i) 
print("\n\n")

# family_ids = [ f"F{id}" for id in range() ]
# dir1 = "../data/sample/F0350/B(외가)/"
# files = os.listdir(f'{dir1}')
# files.sort()
#files.remove('.DS_Store')
# print(files)
for i in pathes:
    full_dir = os.listdir(f'{i}')
    # print(dir3)
    json_files = [ file_name for file_name in full_dir if ( file_name[-4:]=="json" ) ]
    # print(json_files)
    print(f"# of files: {len(json_files)}")
    for indiv_file in json_files:
        with open(f"{i}/{indiv_file}", "r") as st_json:
            st_python = json.load(st_json)
        print(st_python["family_id"], end="\t")
    print('\n')

    #globals()[i.split('.')[0]] = json.load(f'{dir1}/{i}/')