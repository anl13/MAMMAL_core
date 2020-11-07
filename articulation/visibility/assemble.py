import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
idcolors= [
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [255,0,255],
    [255,255,0],
    [0,255,255],
    [128,0,0]
]
def search_id(vec):
    id = -1 
    for i in range(len(idcolors)):
        if idcolors[i][0] == vec[0] and idcolors[i][1] == vec[1] and idcolors[i][2] == vec[2]:
            id = i
            break 
    return id 

def paint():
    img1 = cv2.imread("tex_head_painted.png")
    img1 = img1[:,:,(2,1,0)]
    img2 = cv2.imread("tex_front_leg_painted.png")
    img2 = img2[:,:,(2,1,0)]
    img3 = cv2.imread("tex_back_leg_painted.png") 
    img3 = img3[:,:,(2,1,0)]

    idmap = np.zeros((2048,2048),dtype=np.uint8)
    summap = np.ones((2048,2048,3),dtype=np.uint8) * 255
    for i in range(2048):
        for j in range(2048):
            id = search_id(img1[i,j])
            if id >= 0:
                summap[i,j] = img1[i,j] 
                idmap[i,j] = id+1
                if id == 5: 
                    idmap[i,j] = 18 + 1 
                if id == 6: 
                    idmap[i,j] = 20 + 1
            id = search_id(img2[i,j])
            if id >= 0: 
                idmap[i,j] = id + 5 + 1
                summap[i,j] = img2[i,j]
            id = search_id(img3[i,j])
            if id >= 0: 
                idmap[i,j] = id + 11 + 1
                summap[i,j] = img3[i,j]
    cv2.imwrite("idmap.png", idmap) 

    summap = summap[:,:,(2,1,0)]
    cv2.imwrite("summap.png", summap)

            

if __name__ == "__main__":
    paint() 