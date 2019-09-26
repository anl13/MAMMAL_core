import pickle 
import numpy as np 
import json 
from tqdm import tqdm 

def transfer(inputfile, outfile):
    with open(inputfile,'rb') as f: 
        keypoints = pickle.load(f, encoding='latin1') 
    data = {}
    for key in keypoints.keys():
        label = keypoints[key]
        all_points = {}
        for kpt_idx in range(len(label)):
            points = []
            for i in range(len(label[kpt_idx])):
                points.append(label[kpt_idx][i][0])
                points.append(label[kpt_idx][i][1])
                points.append(label[kpt_idx][i][2])
            all_points.update({str(kpt_idx):points})
        data.update({str(key):all_points})

    with open(outfile, 'w') as f: 
        json.dump(data, f) 


if __name__ == "__main__":
    for frameid in tqdm(range(250)):
        inputfile = "/home/al17/animal/pig/data/keypoints/keypoints_{:06d}.pkl".format(frameid) 
        outfile = "data/keypoints_json/keypoints_{:06d}.json".format(frameid) 
        transfer(inputfile, outfile) 