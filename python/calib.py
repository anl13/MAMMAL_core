import numpy as np  
import cv2 
import os 
from time import time 
import pickle 

from read_xml import get_all_imgs, read_xml_file, preprocess
from read_xml import draw_markers 
from undistortion import undist_points, undist_points_cv2
from matplotlib import pyplot as plt 

CAMIDS = [
    0,1,2,5,6,7,8,9,10,11
]

'''
so, dy/dx = 1.06
'''
def analyze_points(points): 
    # points: 42 * 2 array
    m = points.reshape(7,6,2) 
    # y_d = np.mean((m[:,5,1] - m[:,0,1]) / 5)
    # x_d = np.mean((m[6,:,0] - m[0,:,0]) / 6) 
    lt = m[0,0,:]
    lb = m[0,5,:]
    rt = m[6,0,:]
    rb = m[6,5,:]
    dy = np.linalg.norm(lb - lt) / 5
    dx = np.linalg.norm(rt - lt) / 6 
    dy2 = np.linalg.norm(rb - rt) / 5 
    dx2 = np.linalg.norm(rb - lb) / 6
    from IPython import embed; embed() 

def assume_gt(ratio=1.06):
    gts = np.zeros((42,3),dtype=np.float32)
    for i in range(42):
        r = i % 6 
        c = i // 6 
        x = c * 1.0
        y = r * ratio
        z = 0 
        gts[i,:] = np.array([x,y,z])
    gts = gts * 0.0848
    return gts 


def compute_error(gts, points, rvec, tvec, K, coeff=None):
    R,_ = cv2.Rodrigues(rvec)
    p = gts.dot(R.T) + tvec.squeeze() 
    u = p.dot(K.T)
    u = u / u[:,2:]
    u = u[:,0:2]
    diff = u - points 
    dists = np.linalg.norm(diff, axis=1)
    err = dists.mean()
    return err

def compute_error_cv(gts, points, rvec, tvec, K, coeff):
    projs, _ = cv2.projectPoints(gts, rvec, tvec, K, coeff)
    diff = projs.squeeze() - points 
    dists = np.linalg.norm(diff, axis=1) 
    return dists.mean() 

def search_ratio(points, K, coeff):
    total_errs = []
    for i in range(300):
        ratio = 1.05 + 0.0002 * i 
        gts = assume_gt(ratio)
        retval, rvec, tvec = cv2.solvePnP(gts, points, K, distCoeffs=coeff)
        err = compute_error_cv(gts, points, rvec, tvec, K, coeff)
        pair = [ratio, err]
        total_errs.append(pair)
    total_errs = np.asarray(total_errs) 
    return total_errs

if __name__ == '__main__':
    '''
    load calibration data
    '''
    with open('data/distortion_info.pkl', 'rb') as f: 
        calib_info = pickle.load(f) 
    K = calib_info['K']
    coeff = calib_info['coeff']
    newK = calib_info['newcameramtx'] 
    mapx = calib_info['mapx']
    mapy = calib_info['mapy']
    inv_mapx = calib_info['inv_mapx']
    inv_mapy = calib_info['inv_mapy']

    '''
    load data 
    '''
    imgs = get_all_imgs() 
    marker_map, all_positions = read_xml_file('data/marker-50.xml') 
    all_points = preprocess(marker_map, all_positions)
    np.savetxt('data/newK.txt', newK)
    for imid in range(10):
        img = imgs[imid]
        points = all_points[imid] 
        points = points[0:42,:]
        undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        cv2.imwrite('results/{:02d}.png'.format(CAMIDS[imid]), undist) 
        undist_ps = undist_points_cv2(points, K, coeff, newK)
        np.savetxt('markers/{:02d}.txt'.format(CAMIDS[imid]), undist_ps)

        gts = assume_gt(ratio=1.08575) 
        retval, rvec, tvec = cv2.solvePnP(gts, undist_ps, newK, distCoeffs=None) 
        param = np.vstack([rvec, tvec]) 
        
        print(param)
        np.savetxt('results/{:02d}.txt'.format(CAMIDS[imid]), param)

        '''
        test: search best ratio 
        '''
        # total_errs = search_ratio(undist_ps, newK, None) # optim ratio: 1.0568
        # # total_errs = search_ratio(points, K, coeff)
        # min_index = np.argmin(total_errs[:,1])
        # print(total_errs[min_index])

        # plt.scatter(total_errs[:,0], total_errs[:,1])
        # plt.show()

        # from IPython import embed; embed() 