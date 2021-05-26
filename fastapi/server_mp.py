import io

from segmentation import get_segmentator, get_segments
#from starlette.responses import Response, JSONResponse

#from fastapi import FastAPI, File

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import concurrent
from functools import partial
from PIL import Image
import json
import base64
import numpy as np
import copy
import time
from datetime import datetime


model0 = get_segmentator()
model1 = get_segmentator()
model2 = get_segmentator()
model3 = get_segmentator()
model4 = get_segmentator()
model5 = get_segmentator()
models = [model0, model1, model2, model3, model4, model5]

def get_segmentation_map(file: bytes):
    """Get segmentation maps from image file"""
    tic1 = datetime.now()

    response_dict = dict()

    # 비식별화 
    segmented_img = get_segments(model=models[0], binary_image=file)
    response_dict[0] = np.array(segmented_img).tolist()
    toc_pre = datetime.now()
    print(f"pre model: {toc_pre - tic1}")

    # 각 테스크
    get_segments_for_mp = partial(get_segments, binary_image=copy.deepcopy(file))
    with Pool(5) as pool:
        segmented_images = pool.map(get_segments_for_mp, models[1:])

    for idx, img in enumerate(segmented_images):
        response_dict[idx+1] = np.array(img).tolist()

    toc = datetime.now()
    print(f"multi infer: {toc - toc_pre}")
    print(f"infer total: {toc - tic1}")
    
    return response_dict, toc - tic1

if __name__ == "__main__":

    infer_total_list = []    
    for idx in range(10):
        tic0 = datetime.now()

        img = Image.open("headPoseRight.jpg")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='jpeg')
        img_byte_arr = img_byte_arr.getvalue()

        results, infer_total = get_segmentation_map(img_byte_arr)
        infer_total_list.append(infer_total)

        toc0 = datetime.now()
        print(f"#{idx} total elapsed : {toc0 - tic0}")
    
    print(f"avg infer total: {sum(infer_total_list) / len(infer_total_list)}")
