import torch
from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", 
    revision="fp16", 
    torch_dtype=torch.float16,
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a unicorn on mars"
image = pipe([prompt], num_inference_steps=20).images[0]  
image.save("astronaut_rides_unicorn.png")
