import base64
import io
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from main import instantiate_from_config
import torch

from ldm.models.diffusion.ddim import DDIMSampler

MAX_SIZE = 640

# load safety model
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from imwatermark import WatermarkEncoder
from pydantic import BaseModel
import cv2
import urllib.request
from time import time

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
wm = "StableDiffusionV1-Inpainting"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

app = FastAPI()
sampler = None
model = None
device = None

class SoluteSolvent(BaseModel):
    image: str
    mask: str

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def put_watermark(img):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    return x_checked_image, has_nsfw_concept

def initialize_model(config, ckpt):
    global sampler, device, model
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    if not os.path.exists("inputs_test/"):
        os.mkdir("inputs_test/")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    model = sampler.model

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch

def base64_string_to_pillow_image(base64_str):
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))

def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h//8, w//8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        with torch.autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h//8, w//8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond={"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h//8, w//8]
            samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim+1.0)/2.0,
                                 min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0,2,3,1)
            result, has_nsfw_concept = check_safety(result)
            result = result*255

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]
    result = [put_watermark(img) for img in result]
    return result

@app.post("/inpaint")
async def root(data: SoluteSolvent, s: str):
    response = urllib.request.urlopen(data.image)
    img = Image.open(response)
    response = urllib.request.urlopen(data.mask)
    mask = Image.open(response)
    img.save("inputs_test/img" + str(time()) + ".png")
    mask.save("inputs_test/mask" + str(time()) + ".png")
    
    w, h = img.size
    print(f"loaded input image of size ({w}, {h})")
    if max(w, h) > MAX_SIZE:
        factor = MAX_SIZE / max(w, h)
        w = int(factor*w)
        h = int(factor*h)
    width, height = map(lambda x: x - x % 64, (w, h))  
    image = img.resize((width, height))
    mask = mask.resize((width, height))

    seed = 0
    num_samples = 1
    scale = 7.5
    ddim_steps = 50

    resImg = inpaint(sampler, image, mask, s, seed, scale, ddim_steps, num_samples=num_samples, w=width, h=height)

    resFileName = "outputs/" + str(time()) + "res.png"
    resImg[0].save(resFileName)
    return FileResponse(resFileName)


if __name__ == "__main__":
    initialize_model("configs/stable-diffusion/v1-inpainting-inference.yaml", "model.ckpt")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777, log_level="debug")