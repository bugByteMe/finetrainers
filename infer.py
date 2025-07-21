import torch
from diffusers import WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
import argparse
from transformers import T5TokenizerFast

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
# model_id = "/ocean/model/Wan-AI/Wan2.1-T2V-14B-Diffusers"
# vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
# pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
# pipe.to("cuda")

# prompt = "A cat walks on the grass, realistic"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# output = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     height=480,
#     width=832,
#     num_frames=13,
#     guidance_scale=5.0
# ).frames[0]
# export_to_video(output, "output.mp4", fps=15)

# tokenizer = T5TokenizerFast.from_pretrained("/ocean/model/Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="tokenizer")
# print(tokenizer._convert_id_to_token(6000))
# print(tokenizer("▁Pop", return_tensors="pt", truncation=True, max_length=512,
#             add_special_tokens=False))
# for i in range(6000, 10000):
#     if tokenizer(tokenizer._convert_id_to_token(i), return_tensors="pt", truncation=True, max_length=512,
#                 add_special_tokens=False)['input_ids'].shape[1] == 1:
#         print(f"ID: {i}, Token: {tokenizer._convert_id_to_token(i)}")
# exit(0)
# The video shows a <extra_id_299> car parked in a parking lot.

# prompt = "The video shows a car parked in a parking lot. \
#     The car is positioned at an angle, showcasing its sleek design and black rims.  The logo is visible on the side of the car. \
#         The parking lot appears to be empty, with the car being the main focus of the video. The car's position and the absence of other vehicles suggest that the video might be a promotional or showcase video. \
#             The overall style of the video is simple and straightforward, focusing on the car and its design."
prompt = "A dog. realistic"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video using WanPipeline.")
    parser.add_argument("--num_videos", type=int, default=3, help="Number of videos to generate.")
    parser.add_argument("--output", type=str, default="./", help="Path to the output folder.")

    args = parser.parse_args()

    pipe = WanPipeline.from_pretrained(
        "/ocean/model/Wan-AI/Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16
    ).to("cuda")
    # import safetensors
    # safetensors.torch.load_model(pipe.transformer, "./demo2/output/finetrainers_step_1000/model.safetensors")

    # pipe.load_lora_weights("/home/jrguo/finetrainers/demo2/output/lora_weights/000500", adapter_name="wan-lora")
    # pipe.set_adapters(["wan-lora"], [0.75])

    for i in range(0, args.num_videos, 3):
        videos = pipe(prompt, negative_prompt=negative_prompt, num_videos_per_prompt=3, num_frames=13).frames
        for j, video in enumerate(videos):
            output_path = f"{args.output}/video_{i + j}.mp4"
            print(f"Saving video {i + j} to {output_path}")
            export_to_video(video, output_path, fps=8)