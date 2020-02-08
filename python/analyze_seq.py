import numpy as np 
import cv2 
import matplotlib.pyplot as plt  
import pickle
from mpl_toolkits.mplot3d import Axes3D


def test():
    # data = [[],[],[],[]]
    # for id in range(4):
    #     for frame in range(5000):
    #         filename = "E:/pig_results/state_{:d}_{:06d}.pig".format(id, frame)
    #         framedata = np.loadtxt(filename) 
    #         data[id].append(framedata[-9:])
    # with open("seq.pkl", 'wb') as f: 
    #     pickle.dump(data, f, protocol=2) 

    with open("seq.pkl", "rb") as f: 
        data = pickle.load(f, encoding="latin1")

    data = np.asarray(data)  
    centers = data[3,:,3:6]

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(centers[:,0], centers[:,1], zs=centers[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    from IPython import embed; embed()

if __name__ == "__main__":
    test() 