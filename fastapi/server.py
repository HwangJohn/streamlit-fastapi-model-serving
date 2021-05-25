import io

from segmentation import get_segmentator, get_segments
from starlette.responses import Response, JSONResponse

from fastapi import FastAPI, File

import multiprocessing as mp
from functools import partial
from PIL import Image
import json
import base64
import numpy as np

model0 = get_segmentator()
model1 = get_segmentator()
model2 = get_segmentator()
model3 = get_segmentator()
model4 = get_segmentator()
model5 = get_segmentator()
models = [model0, model1, model2, model3, model4, model5]

app = FastAPI(
    title="DeepLabV3 image segmentation",
    description="""Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)


@app.post("/segmentation")
def get_segmentation_map(file: bytes = File(...)):
    """Get segmentation maps from image file"""

    response_dict = dict()

    # 비식별화 
    segmented_img = get_segments(model=models[0], binary_image=file)
    response_dict[0] = np.array(segmented_img).tolist()

    # 각 테스크
    get_segments_for_mp = partial(get_segments, binary_image=file)
    with mp.Pool() as p:
        segmented_images = p.map(get_segments_for_mp, models[1:])

    for idx, img in enumerate(segmented_images):
        response_dict[idx+1] = np.array(img).tolist()

    return JSONResponse(json.dumps(response_dict), media_type="application/json")

    # bytes_io = io.BytesIO()
    # segmented_image.save(bytes_io, format="PNG")
    # return Response(bytes_io.getvalue(), media_type="image/png")

if __name__ == "__main__":
    img = Image.open("headPoseRight.jpg")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    # result = get_segments(model0, img_byte_arr)
    results = get_segmentation_map(img_byte_arr)

    result_dict = json.loads(json.loads(results.body.decode()))
    print(result_dict.keys())

    # print(img_byte_arr)
    # get_segmentation_map()