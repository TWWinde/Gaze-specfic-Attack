import os
import cv2
import sys
import glob
import argparse
import h5py
import numpy as np
import tqdm
sys.path.append('../gaze_estimation')

'''
    This file transfers original .avi of lpw dataset to .h5 file
'''

parser = argparse.ArgumentParser()
parser.add_argument('--noDisp', help='Specify flag to display labelled images', type=int, default=1)
parser.add_argument('--path2ds',
                    help='Path to dataset',
                    type=str,
                    default='./data')
args = parser.parse_args()
PATH_DIR = os.path.join(args.path2ds, 'LPW')
PATH_DS = os.path.join(args.path2ds)
PATH_MASTER = os.path.join(args.path2ds, 'MasterKey')
list_ds = list(os.walk(PATH_DIR))[0][1]

print('Extracting LPW')


def generateEmptyStorage():
    '''
    This file generates an empty dictionary with
    all relevant fields. This helps in maintaining
    consistency across all datasets.
    '''
    Data = {k: [] for k in ['Images', 'pupil_loc', 'Name']}
    Data['Images'] = []
    Data['pupil_loc'] = []
    Data['Name'] = []
    return Data


def readFormattedText(path2file, ignoreLines):
    data = []
    count = 0
    f = open(path2file, 'r')
    for line in f:
        d = [float(d) for d in line.split()]
        count = count + 1
        if d and count > ignoreLines:
            data.append(d)
    f.close()
    return data


def add_data_to_hdf5(person_id, images, pupil_loc_list, output_path):

    with h5py.File(output_path, 'a') as f_output:
        for index, (image, pupil_loc) in tqdm.tqdm(enumerate(zip(images, pupil_loc_list)), leave=False):
            f_output.create_dataset(f'{person_id}/image/{index:04}', data=image)
            f_output.create_dataset(f'{person_id}/pupil_loc/{index:04}', data=pupil_loc)



if __name__ == '__main__':
    output_path = os.path.join(args.path2ds, 'LPW.h5')

    for name in sorted(list_ds):  # name of subfiles
        opts = glob.glob(os.path.join(PATH_DIR, name, '*.avi'))
        Images = []
        pupil_loc_list = []
        print(f'start {name}')
        for Path2vid in opts:
            loc, fName = os.path.split(Path2vid)
            fName = os.path.splitext(fName)[0]
            Path2text = os.path.join(loc, fName + '.txt')
            PupilData = np.array(readFormattedText(Path2text, 0))
            VidObj = cv2.VideoCapture(Path2vid)

            fr_num = 0
            while VidObj.isOpened():
                ret, I = VidObj.read()
                if ret:
                    Images.append(I)
                    pupil_loc = PupilData[fr_num, :]
                    pupil_loc_list.append(pupil_loc)
                    fr_num += 1
                else:
                    break
        person_id = f'p{int(name):02}'
        print(person_id)
        add_data_to_hdf5(person_id, Images, pupil_loc_list, output_path)

