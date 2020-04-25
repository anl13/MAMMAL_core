import numpy as np 
import cv2 
import matplotlib.pyplot as plt  
import pickle
import json 

CAMIDS = [
   0, 1,2,5,6,7,8,9,10,11
]

def readbox(frameid):
    path = "E:/sequences/20190704_morning/boxes_mm/boxes_{:06d}.json".format(frameid)
    with open(path, 'r') as f: 
        data = json.load(f)
    return data 
    from IPython import embed; embed() 

def compute_background(camid):
    folder = "E:/sequences/20190704_morning/images/cam" + str(camid) + "/"
    img = np.zeros([1080,1920,3],np.float64)
    weights = np.zeros([1080,1920],np.float64)
    frames = range(0,10000,10)
    for i in frames:
        boxes = readbox(i)
        box = boxes[str(camid)]
        name = "{}{:06d}.jpg".format(folder,i)
        current_img = cv2.imread(name)
        w = np.ones([1080,1920],np.uint8)
        for b in box:
            cv2.rectangle(current_img,(int(b[0]),int(b[1]),int(b[2]-b[0]),int(b[3]-b[1])),(0,0,0),-1)
            cv2.rectangle(w,(int(b[0]),int(b[1]),int(b[2]-b[0]),int(b[3]-b[1])),0,-1)
        weights = weights + w.astype(np.float64)
        img = img + current_img.astype(np.float64)
        print(i)

    img = img / weights.reshape([1080,1920,1])
    img2 = img.astype(np.uint8)
    cv2.imwrite("bg{}.png".format(camid), img2)

if __name__ == "__main__":
    for i in CAMIDS:
        compute_background(i)