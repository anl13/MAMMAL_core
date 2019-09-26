import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm   
import cv2 

rgb = cm.get_cmap(plt.get_cmap('jet'))(np.linspace(0.0,1.0,256))[:,:3]
bgr = rgb[:,(2,1,0)]

RGB_256_CM = rgb.copy() 
BGR_256_CM = bgr.copy() 
