import cv2 
import numpy as np 

def extract():
    filename = "result_data/reproj_topdown.avi" 
    outname = "result_data/reproj_topdown_cam0.avi"
    fourcc = cv2.VideoWriter.fourcc('M','P','E','G')
    writer = cv2.VideoWriter(outname, fourcc, 25.0, (1920,1080)) 
    cap = cv2.VideoCapture(filename)

    for i in range(1500):
        rt, img = cap.read()
        if not rt: 
            break 
        img2 = img[0:1080,0:1920,:]
        writer.write(img2) 
    
    cap.release()
    writer.release() 

def convert_render():
    filename = "result_data/render.avi" 
    outname  = "result_data/render_bgr.avi"
    fourcc = cv2.VideoWriter.fourcc('M','P','E','G')
    writer = cv2.VideoWriter(outname, fourcc, 25.0, (1024,1024)) 
    cap  = cv2.VideoCapture(filename) 
    for i in range(1500):
        rt, img = cap.read() 
        if not rt: 
            break 
        img2 = img[:,:,(2,1,0)]
        writer.write(img2) 
    cap.release() 
    writer.release() 

if __name__ == "__main__":
    extract() 
    # convert_render() 