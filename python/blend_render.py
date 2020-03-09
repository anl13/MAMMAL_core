import numpy as np   
import cv2  

# img: [h,w,c]
def blend(img1, img2, alpha):
    # from IPython import embed; embed() 
    # a = img2 < 255
    img = img1.copy()
    img = img1 * alpha + img2 * (1-alpha)
    return img 

def blend_frame(id):
    folder = "E:/debug_pig/render/"
    img1_file = folder + "raw_{:06d}.jpg".format(id)
    img2_file = folder + "{:06d}.jpg".format(id) 
    img1 = cv2.imread(img1_file) 
    img2 = cv2.imread(img2_file) 
    img = blend(img1, img2, 0.5)
    img_file = "E:/debug_pig/blend/{:06d}.jpg".format(id) 
    cv2.imwrite(img_file, img) 

if __name__ == "__main__":
    for i in range(6330,6390):
        blend_frame(i)