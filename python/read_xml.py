import numpy as np  
import xml.etree.ElementTree as ET 
import os 
import cv2 

def read_xml_file(filename):
    tree = ET.parse(filename) 
    document = tree.getroot() 
    chunk = document[0] 
    markers = chunk[2] 
    frames = chunk[7] 
    # parse markers 
    marker_list = []
    for child in markers.getchildren():
        mid = int(child.attrib['id'])
        mlabel = int(child.attrib['label'][6:])
        marker_list.append({
            'id': mid, 
            'label': mlabel
        })
    # parse marker position in each frame 
    positions = frames[0][0]
    all_positions = []
    for marker in positions.getchildren(): 
        id = marker.attrib['marker_id']
        locations = []
        for location in marker.getchildren():
            camid = location.attrib['camera_id']
            x = location.attrib['x']
            y = location.attrib['y'] 
            locations.append({
                'camid': int(camid), 
                'x': float(x), 
                'y': float(y)
            })
        all_positions.append([id, locations])
    return marker_list, all_positions 

def preprocess(marker_list, all_positions): 
    mark_num = len(all_positions)
    frames = np.zeros((10, mark_num, 2), np.float32) 
    for i in range(mark_num):
        _, positions = all_positions[i]
        for pos in positions:
            camid = pos['camid']
            x = pos['x']
            y = pos['y']
            frames[camid, i, 0] = x 
            frames[camid, i, 1] = y 
    return frames.astype(np.float32)

def get_all_imgs():
    folder = '../data/calib_1_color' 
    img_names = sorted(os.listdir(folder)) 
    imgs = []
    for i in range(len(img_names)):
        name = os.path.join(folder, img_names[i])
        img = cv2.imread(name) 
        # img_rgb = img[:,:,(2,1,0)]
        # imgs.append(img_rgb)
        imgs.append(img)
    return imgs 

# markers: markers of an image
def draw_markers(img, markers):
    for i in range(42):
        marker = markers[i] 
        cv2.circle(img, center=tuple(marker), radius=10, color=(0,0,255), thickness=-1)

def write_markers_txt(markers):
    for i in range(10):
        outname = "markers/{:02d}.txt".format(i) 
        np.savetxt(outname, markers[i][:42,:])

def analyze_markers(points): 
    # points: 42 * 2 array
    m = points.reshape(7,6,2) 
    y_d = np.mean((m[:,5,1] - m[:,0,1]) / 5)
    x_d = np.mean((m[6,:,0] - m[0,:,0]) / 6) 
    from IPython import embed; embed() 

if __name__ == "__main__":
    filename = 'marker-50.xml'
    marker_map, all_positions = read_xml_file(filename) 
    points = preprocess(marker_map, all_positions)
    images = get_all_imgs() 
    # draw_markers(images[3], points[3]) 

    # write_markers_txt(points)
    analyze_markers(points[3, 0:42, :]) 