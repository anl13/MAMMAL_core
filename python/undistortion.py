import numpy as np 
import cv2 
import os 

from read_xml import get_all_imgs, read_xml_file, preprocess
from read_xml import draw_markers 
from time import time
import pickle 

CAMIDS = [
    0,1,2,5,6,7,8,9,10,11
]
def undistort_image(img, mapx, mapy):
    h,w,c = img.shape 
    im_out = img.copy() 
    for i in range(h):
        for j in range(w):
            x = mapx[i,j]
            y = mapy[i,j]
            x1 = int(x)
            x2 = x1 + 1
            y1 = int(y) 
            y2 = y1 + 1 
            if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h: 
                im_out[i,j] = np.array([0,0,0])
                continue 
            dx = x - x1 
            dy = y - y1 
            lt = img[y1,x1]
            rt = img[y1,x2]
            lb = img[y2,x1] 
            rb = img[y2,x2] 
            out = lt*(1-dx)*(1-dy) + rt*dx*(1-dy) + lb*(1-dx)*dy + rb*dx*dy 
            im_out[i,j] = out 
    return im_out 

'''
input: 
  mapx: mapx [h,w] 
  mapy: mapy [h,w]
return: 
  result: [w*h, 5], nearest neighbor indices 
  dists : [w*h, 5], nearest neighbor distances 
'''
def nn_search(mapx, mapy): 
    # build dataset 
    h = mapx.shape[0] 
    w = mapx.shape[1]
    dataset = np.zeros((w*h, 2))
    dataset[:,0] = mapx.reshape((w*h,1)).squeeze() 
    dataset[:,1] = mapy.reshape((w*h,1)).squeeze()  # index = hi * w + wi
    # build kd-tree
    from pyflann import FLANN
    flann = FLANN() 
    params = flann.build_index(dataset, algorithm='kdtree', trees=4)
    testset = np.zeros((w*h,2)) 
    for i in range(h):
        for j in range(w):
            testset[w*i+j,:] = np.array([j,i]) 
    print('created matrix')
    # query nearest neighbours
    start = time() 
    result, dists = flann.nn_index(testset, 5, checks = params['checks'])
    end = time() 
    print("elapsed {} s".format(end-start))
    data = {} 
    data['result'] = result
    data['dists'] = dists
    with open('data/inverse_map_dict.pkl', 'wb') as f: 
        pickle.dump(data, f, protocol=2)
    return result, dists

def inverse_dist_weighting(xs, ds): 
    assert(xs.shape == ds.shape)
    ws = 1 / ds 
    interp = (xs * ws).sum() / (ws.sum())
    return interp 

def get_inverse_remap(result, dists, w, h):
    inv_mapx = np.zeros((h,w), dtype=np.float32)
    inv_mapy = np.zeros((h,w), dtype=np.float32)
    for i in range(h): 
        for j in range(w): 
            index    = i * w + j 
            nn_inds  = result[index] 
            nn_inds_x = (nn_inds % w).astype(np.int)
            nn_inds_y = (nn_inds / w).astype(np.int)
            nn_dists = dists[index]
            new_x    = inverse_dist_weighting(nn_inds_x, nn_dists)
            new_y    = inverse_dist_weighting(nn_inds_y, nn_dists)
            inv_mapx[i,j] = new_x 
            inv_mapy[i,j] = new_y 
    return inv_mapx, inv_mapy 

def linear_interp(y,x,map):
    assert(y>=0 and x>=0 and y<=map.shape[0]-1 and x<=map.shape[1]-1)
    y1 = int(y) 
    y2 = y1 + 1 
    x1 = int(x) 
    x2 = x1 + 1 
    dx = x - x1 
    dy = y - y1 
    value = map[y1,x1] * (1-dy) * (1-dx) \
         + map[y1,x2] * (1-dy) * dx \
         + map[y2,x1] * dy * (1-dx) \
         + map[y2,x2] * dy * dx 
    return value 

def undist_point(point, inv_mapx, inv_mapy): 
    [x,y] = point 
    new_x = linear_interp(y,x,inv_mapx) 
    new_y = linear_interp(y,x,inv_mapy) 
    return np.array([new_x, new_y])

# points are in [x,y] format, as used in cv2
def undist_points(points, inv_mapx, inv_mapy): 
    new_points = points.copy() 
    for i in range(points.shape[0]): 
        new_points[i] = undist_point(points[i], inv_mapx, inv_mapy)
    return new_points


'''
points: [N,2]
'''
def undist_points_cv2(points, K, coeff, newcameramtx):
    points_cv = points.copy() 
    points_cv = points_cv.reshape([points_cv.shape[0], 1, points_cv.shape[1]])
    new_points_2 = cv2.undistortPoints(points_cv, K, coeff, P=newcameramtx) 
    new_points_2 = new_points_2.squeeze() 
    return new_points_2 # [N,2]

if __name__ == "__main__":
    '''
    basic intrinsic calibration information
    '''
    K = [[1625.30923, 0, 963.88710], 
         [0, 1625.34802, 523.45901], 
         [0,      0,     1]] 
    K = np.array(K, dtype=np.float32)
    coeff = [-0.35582, 0.14595, -0.00031, -0.00004, 0.00000]
    coeff = np.array(coeff, dtype=np.float32)
    w = 1920 
    h = 1080 
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, coeff, (w,h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, coeff, None, newcameramtx, (w,h), 5)

    '''
    get original markers 
    '''
    filename = 'data/marker-50.xml'
    marker_map, all_positions = read_xml_file(filename) 
    all_points = preprocess(marker_map, all_positions)
    points = all_points[0] 
    imgs = get_all_imgs() 

    '''
    compute nearest neighbours 
    '''
    # nn_search(mapx, mapy) 

    '''
    compute and save inverse distortion mapping
    '''
    if os.path.exists('data/distortion_info.pkl'):
        with open('data/distortion_info.pkl','rb') as f: 
            calib_info = pickle.load(f) 
        inv_mapx = calib_info['inv_mapx']
        inv_mapy = calib_info['inv_mapy']
    else: 
        with open('data/inverse_map_dict.pkl', 'rb') as f: 
            data = pickle.load(f) 
        result = data['result']
        dists = data['dists']
        inv_mapx, inv_mapy = get_inverse_remap(result, dists, w, h) 
        calib_info = {}
        calib_info['K'] = K 
        calib_info['coeff'] = coeff 
        calib_info['type'] = 1
        calib_info['mapx'] = mapx
        calib_info['mapy'] = mapy 
        calib_info['inv_mapx'] = inv_mapx 
        calib_info['inv_mapy'] = inv_mapy 
        calib_info['newcameramtx'] = newcameramtx
        with open('data/distortion_info.pkl', 'wb') as f: 
            pickle.dump(calib_info, f, protocol=2) 

    '''
    test inverse mapping
    '''
    img = imgs[0].copy()
    undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    redist = cv2.remap(undist, inv_mapx, inv_mapy, cv2.INTER_LINEAR) 
     
    new_points_2 = undist_points_cv2(points, K, coeff, newcameramtx)
    new_points = undist_points(points, inv_mapx, inv_mapy)
    draw_markers(undist, new_points)

    

    cv2.namedWindow('undist', cv2.WINDOW_NORMAL)
    cv2.imshow('undist', undist)
    cv2.namedWindow("redist", cv2.WINDOW_NORMAL)
    cv2.imshow("redist", redist) 
    cv2.imwrite('data/undist.png', undist)
    cv2.imwrite('data/redist.png', redist)
    cv2.waitKey() 
    cv2.destroyAllWindows()

    from IPython import embed; embed() 