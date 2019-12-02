import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm   
import cv2 

rgb = cm.get_cmap(plt.get_cmap('jet'))(np.linspace(0.0,1.0,256))[:,:3]
bgr = rgb[:,(2,1,0)]

RGB_256_CM = rgb.copy() 
BGR_256_CM = bgr.copy() 

# https://www.rapidtables.com/web/color/RGB_Color.html


def save_a_map():
    filename = "/home/al17/animal/animal_calib/data/colormaps/jet.txt"
    colormap_int = (RGB_256_CM * 256).astype(np.int)
    np.savetxt(filename, colormap_int, fmt='%-d') 
    from IPython import embed; embed() 

def save_matlab_map(cm_type, length):
    filename = "/home/al17/animal/animal_calib/data/colormaps/"+cm_type+".txt"
    rgb = cm.get_cmap(plt.get_cmap(cm_type))(np.linspace(0.0,1.0,length))[:,:3]
    colormap_int = (rgb * 256).astype(np.int)
    np.savetxt(filename, colormap_int, fmt='%-d')   


if __name__ == "__main__":
    save_matlab_map("gist_ncar", 128)