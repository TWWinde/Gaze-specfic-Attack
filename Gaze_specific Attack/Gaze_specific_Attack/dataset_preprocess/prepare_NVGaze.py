import os
import cv2
import sys
import argparse
import h5py
import numpy as np
import tqdm
sys.path.append('../gaze_estimation')


'''
    This file transfers original NVGaze dataset to .h5 file
'''

parser = argparse.ArgumentParser()
parser.add_argument('--noDisp', help='Specify flag to display labelled images', type=int, default=1)
parser.add_argument('--path2ds',
                    help='Path to dataset',
                    type=str,
                    default='./data')
args = parser.parse_args()

PATH_DIR = os.path.join(args.path2ds, 'NVGaze')

print('Extracting NVGaze')

Image_counter = 0.0
ds_num = 0


def readFormattedText(path2file, ignoreLines):
    data = []
    count = 0
    f = open(path2file, 'r')
    for line in f:
        if line is not None and count > ignoreLines:
            d = [d.strip() for d in line.split(',')]
            data.append(d)
        count = count + 1
    f.close()
    return data


def add_data_to_hdf5(person_id, images, gazes, eyes, output_path):
    with h5py.File(output_path, 'a') as f_output:
        for index, (image, gaze, eye) in tqdm.tqdm(enumerate(zip(images, gazes, eyes)), leave=False):
            f_output.create_dataset(f'{person_id}/image/{index:06}', data=image)
            f_output.create_dataset(f'{person_id}/gaze/{index:06}', data=gaze)
            f_output.create_dataset(f'{person_id}/eye/{index:06}', data=eye)


output_path = os.path.join(args.path2ds, 'NVGaze.h5')

for num in range(1, 15):  # name of subfiles
    image_path = os.path.join(PATH_DIR, f'{num:02}')
    Path2text = os.path.join(PATH_DIR, f'{num:02}.csv')
    data = readFormattedText(Path2text, 13)
    Images = []
    gazes = []
    eye = []
    n=1
    print(f'start {num}')
    for line in data:
        print(line[0])
        I = cv2.imread(os.path.join(image_path, line[0]))
        Images.append(I)
        eye.append(line[1])
        gaze = [float(item) for item in line[2:4]]
        gazes.append(gaze)
        n+=1
        if n==3000:
            break
    person_id = f'p{int(num):02}'
    print(person_id)
    gazes = np.array(gazes)
    add_data_to_hdf5(person_id, Images, gazes, eye, output_path)
