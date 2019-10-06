import numpy as np 
import json 

def convert(inpath, outpath):
    with open(inpath, 'r') as f: 
        data = json.load(f)

    data = np.asarray(data['pigs'])

    from IPython import embed; embed() 

if __name__ == "__main__":
    inpath = "/home/al17/animal/animal_calib/build/results/json/003732.json"
    outpath = "/home/al17"
    convert()
