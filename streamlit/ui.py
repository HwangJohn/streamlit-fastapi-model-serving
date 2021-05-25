import io

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

import json

# interact with FastAPI endpoint
backend = "http://fastapi:8000/segmentation"


def process(image, server_url: str):

    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return r


# construct UI layout
st.title("DeepLabV3 image segmentation")

st.write(
    """Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8000/docs` for FastAPI documentation."""
)  # description and instructions

input_image = st.file_uploader("insert image")  # image upload widget

if st.button("Get segmentation map"):

    col11, col12 = st.beta_columns(2)
    col21, col22 = st.beta_columns(2)
    col31, col32 = st.beta_columns(2)
    col41, col42 = st.beta_columns(2)
    col51, col52 = st.beta_columns(2)
    col61, col62 = st.beta_columns(2)

    if input_image:
        segments = process(input_image, backend)
        segmented_images = json.loads(segments.body.decode())
        original_image = Image.open(input_image).convert("RGB")
        segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
        col11.header("Original")
        col11.image(original_image, use_column_width=True)
        col12.header("Segmented")
        col12.image(segmented_images["0"], use_column_width=True)
        col21.header("Original")
        col21.image(original_image, use_column_width=True)
        col22.header("Segmented")
        col22.image(segmented_images["1"], use_column_width=True)
        col31.header("Original")
        col31.image(original_image, use_column_width=True)
        col32.header("Segmented")
        col32.image(segmented_images["2"], use_column_width=True)
        col41.header("Original")
        col41.image(original_image, use_column_width=True)
        col42.header("Segmented")
        col42.image(segmented_images["3"], use_column_width=True)
        col51.header("Original")
        col51.image(original_image, use_column_width=True)
        col52.header("Segmented")
        col52.image(segmented_images["4"], use_column_width=True)
        col61.header("Original")
        col61.image(original_image, use_column_width=True)
        col62.header("Segmented")
        col62.image(segmented_images["5"], use_column_width=True)                                               

    else:
        # handle case with no image
        st.write("Insert an image!")
