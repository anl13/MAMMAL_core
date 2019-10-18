import numpy as np 
import cv2 
import os 

## conclusion 20191018 AN Liang
# Because pig has large white area on its body, where no flow can be estimated, 
# optical flow methods does not suitable for pig center tracking. 

def get_image_names(folder):
    names = os.listdir(folder)
    clean_names = []
    for name in names:
        if "png" in name or "jpg" in name: 
            clean_names.append(name) 
    clean_names.sort() 
    return clean_names

def compute_flow(_img1, _img2):
    im1 = _img1.copy() 
    im2 = _img2.copy() 
    grey1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) 
    hsv = np.zeros_like(im1)  
    hsv[...,1] = 255
    grey2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(grey1, grey2, None, 0.5, 5, 25, 5, 5, 1.2, 0) 
    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) 
    hsv[...,0] = ang*180/np.pi/2 
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) 
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 

    cv2.namedWindow("flow", cv2.WINDOW_NORMAL) 
    cv2.imshow("flow", rgb ) 
    cv2.namedWindow("raw1", cv2.WINDOW_NORMAL) 
    cv2.namedWindow("raw2", cv2.WINDOW_NORMAL) 
    cv2.imshow("raw1", im1) 
    cv2.imshow("raw2", im2) 
    key = cv2.waitKey(20) 
    if key == 27:
        exit()
    # cv2.destroyAllWindows() 
    
def main():
    folder = "/home/al17/animal/pig-data/sync/0819/sync_morning1/cam0/"
    names = get_image_names(folder)

    for i in range(999):
        im1 = cv2.imread(folder + names[i]) 
        im2 = cv2.imread(folder + names[i+1])
        compute_flow(im1, im2)


if __name__ == "__main__":
    main() 