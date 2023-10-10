import torch
import csv
from torch import nn
import numpy as np
import scipy.io
import os
import sys
from sys import platform
from datetime import datetime
import cv2
import pandas as pd

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

annotations = scipy.io.loadmat('./mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat', struct_as_record=False)

release = annotations['RELEASE']

must_be_list_fields = ["annolist", "annorect", "point", "img_train", "single_person", "act", "video_list"]

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/openpose/python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            #sys.path.append('./openpose/python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # # Flags
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    # args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../openpose/models/"

    
    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
except Exception as e:
    print(e.with_traceback())
    sys.exit(-1)
    
    
    
def get_points(img):
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.poseKeypoints[0], datum.cvOutputData

def generate_dataset_obj(obj):
    if type(obj) == np.ndarray:
        dim = obj.shape[0]
        if dim == 1:
            ret = generate_dataset_obj(obj[0])
        else:
            ret = []
            for i in range(dim):
                ret.append(generate_dataset_obj(obj[i]))

    elif type(obj) == scipy.io.matlab.mio5_params.mat_struct:
        ret = {}
        for field_name in obj._fieldnames:
            field = generate_dataset_obj(obj.__dict__[field_name])
            if field_name in must_be_list_fields and type(field) != list:
                field = [field]
            ret[field_name] = field

    else:
        ret = obj

    return ret

def print_dataset_obj(obj, depth = 0, maxIterInArray = 20):
    prefix = "  "*depth
    if type(obj) == dict:
        for key in obj.keys():
            print("{}{}".format(prefix, key))
            print_dataset_obj(obj[key], depth + 1)
    elif type(obj) == list:
        for i, value in enumerate(obj):
            if i >= maxIterInArray:
                break
            print("{}{}".format(prefix, i))
            print_dataset_obj(value, depth + 1)
    else:
        print("{}{}".format(prefix, obj))

# Convert to dict
dataset_obj = generate_dataset_obj(release)

data = dataset_obj['annolist']

training_data = []

#generate training dataset
test_data = []


for i in range(len(data)):
    d = dataset_obj['annolist'][i]
    name = d['image']['name']

    try:
        img = cv2.imread("mpii_human_pose_v1/images/"+name)
        points, cv_output_data = get_points(img)
        
        # 2d image so remove the third dimension
        
        points = points[:, :2]
        label = dataset_obj['act'][i]['act_id']
        
        
        # convert points to tensor
        points = points.astype(np.int16)
        pointsStr =  "[" + ", ".join([str(p) for p in points.flatten()]) + "]"
        
        if dataset_obj['img_train'][i] == 0:
            training_data.append((name, label, pointsStr))
            print("creating training data for {}".format(name))
        elif dataset_obj['img_train'][i] == 1:
            test_data.append((name, label, pointsStr))
            print("creating test data for {}".format(name))
        
        
        
    except Exception as e:
        continue
    
    
pd.DataFrame(training_data).to_csv('training_data1.csv', index=False)

    
# with open('training_data.csv','w') as out:
#     csv_out=csv.writer(out)
#     csv_out.writerows(training_data)

# generate test dataset



with open('test_data.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerows(test_data)
# print(training_data)

# initalize model

#three leaky relu layers and a hyperbolic tangent layer

