import io

import torch
from PIL import Image
from torchvision import transforms

from einops import rearrange
import numpy as np

# adapted from https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/


def get_segmentator():

    model = torch.hub.load(
        "pytorch/vision:v0.6.0", "deeplabv3_resnet101", pretrained=True
    )
    model.eval()

    return model


def get_segments(model=None, binary_image=None, max_size=512):
    print("segmenting")
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    # resized_image = input_image.resize(
    #     (
    #         int(input_image.width * resize_factor),
    #         int(input_image.height * resize_factor),
    #     )
    # )
    resized_image = input_image.resize((768, 768))

    np_resized_image = np.array(resized_image)
    np_fetched_images = rearrange(np_resized_image, "(h1 h2) (w1 w2) C -> (h1 w1) h2 w2 C", h1=3, w1=3)
    # np_fetched_image = np_fetched_image.transpose(0,3,1,2)

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # input_tensor = preprocess(resized_image)
    # input_batch = input_tensor.unsqueeze(
    #     0 
    # )  # create a mini-batch as expected by the model

    input_batchs = []
    for np_img in np_fetched_images:
        input_batchs.append(preprocess(np_img))
    t_input_batchs = torch.stack(input_batchs)

    with torch.no_grad():
        output = model(t_input_batchs)["out"]

    output_predictions = output.argmax(1)

    # create a color palette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    output_predictions = rearrange(output_predictions, '(b1 b2) H W -> (b1 H) (b2 W)', b1=3, b2=3)  

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(
        input_image.size
    )
    r.putpalette(colors)

    return r
