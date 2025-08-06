import torch
from finetrainers.models.wan.inv_specification import WanPipelineInv
from diffusers.utils import export_to_video
import safetensors
import torchvision
import numpy as np

class EmbeddingWrap(torch.nn.Module):
    """
    A simple wrapper for the embedding to be trained.
    This is used to ensure that the embedding is treated as a single module
    and can be easily moved to the correct device.
    """

    def __init__(self, embedding: torch.nn.Module):
        super().__init__()
        self.embedding = torch.nn.Parameter(embedding)

    def forward(self, *args, **kwargs):
        return self.embedding(*args, **kwargs)

def tokenize(text, tokenizer):
    text_ids = tokenizer(
        text,
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    print(f"Tokenized text: {text_ids['input_ids']}")
    # Reverse the tokenization to get the text tokens
    text_tokens = tokenizer.convert_ids_to_tokens(text_ids["input_ids"][0])
    print(f"Text tokens: {text_tokens}")
    return text_tokens

def load_pipeline(pretrained_model_path : str, lora_weight_path = None, embedding_path = None) -> WanPipelineInv:
    pipe = WanPipelineInv.from_pretrained(
        pretrained_model_path, torch_dtype=torch.bfloat16
    ).to("cuda")

    if lora_weight_path is not None:
        pipe.load_lora_weights(lora_weight_path, adapter_name="wan-lora")
        pipe.set_adapters(["wan-lora"], [1.0])

    embedding = None
    if embedding_path is not None:
        # Load the embedding from a file
        embedding_tensor = torch.Tensor(1, 1, 4096)
        embedding = EmbeddingWrap(embedding_tensor)
        safetensors.torch.load_model(embedding, embedding_path)

    return pipe, embedding

def save_attn_map(attn_map, video_frame, step, seed, p_idx, p, j):
    attn_map = attn_map[0, :, :10]
    attn_map = attn_map.reshape(
        4, 30, 52, 10
    )[0].permute(2, 0, 1).to(torch.float32).cpu().numpy() # N, H, W
    tokens = tokenize(p, pipe.tokenizer)
    for k, attn_img in enumerate(attn_map):
        if k >= len(tokens):
            break
        print(f"Min: {attn_img.min()}, Max: {attn_img.max()}")
        # Normalize the attention map to [0, 1]
        attn_img = (attn_img - attn_img.mean()) / (attn_img.std() + 1e-6)
        # attn_img = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min())

        # Save attention map as an image
        attn_img_path = f"./it_{step}_s_{seed}_p_{p_idx}_B_{j}_amap_{k}_{tokens[k]}.png"
        print(f"Saving attention map {k} to {attn_img_path}")

        H, W, C = video_frame.shape

        # Resize map to match video frame size
        attn_img = torchvision.transforms.functional.resize(
            torch.tensor(attn_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            (H, W)
        ).squeeze(0).squeeze(0).numpy()

        # Convert video_frame to float32 for computation
        video_frame_float = video_frame.astype(np.float32) / 255.0 if video_frame.dtype == np.uint8 else video_frame.copy()
        video_frame_annotated = 0.1 * video_frame_float + 0.9 * video_frame_float * attn_img[..., None]
        # clip to [0, 1]
        video_frame_annotated = np.clip(video_frame_annotated, 0, 1)
        # Cat horizontal (ensure both have same dtype)
        combined = np.concatenate(
            [video_frame_float, video_frame_annotated], axis=0
        )
        print(f"Combined shape: {combined.shape}")
        torchvision.utils.save_image(torch.tensor(combined, dtype=torch.float32).permute(2, 0, 1), attn_img_path)

pretrained_model_path = "/ocean/model/Wan-AI/Wan2.1-T2V-14B-Diffusers"
lora_weight_path = "/home/jrguo/finetrainers/demo/output-sled-dog-joint-p2-4e-4/lora_weights/002000"
embedding_path = "./demo/output-sled-dog-joint-p2-4e-4/finetrainers_step_2000/model_1.safetensors"
dump_attention_maps = False

pipe, embedding = load_pipeline(pretrained_model_path, lora_weight_path, embedding_path)

# tokenize("A * dog running in the snow", pipe.tokenizer)

map_storage = []
if dump_attention_maps:
    from attn_hook import WanAttnProcessor2_0_Hook
    pipe.transformer.blocks[39].attn2.processor = WanAttnProcessor2_0_Hook(map_storage)

seeds = [12, 42, 62]
step = 2000
# prompts = [
#     "A * dog running in the snow.",
#     "A dog running in the snow.",
#     "A * dog at the beach.",
#     "A dog in * appearance with sunglasses at the beach.",
#     "A dog with sunglasses at the beach."
# ]
prompts = [
    "A dog in * appearance running in the snow.",
    "A dog running in the snow.",
    "A dog in * appearance at the beach.",
    "A dog in * appearance with sunglasses at the beach.",
    "A dog with sunglasses at the beach."
]
# seeds = [12]
# step = 2000
# prompts = [
#     "A * dog",
# ]
for seed in seeds:
    generator = torch.Generator(device=pipe.transformer.device).manual_seed(seed)
    for p_idx, p in enumerate(prompts):
        # Clear storage
        map_storage.clear()
        # videos, attn_outputs = pipe(p, negative_prompt="", num_videos_per_prompt=2, num_frames=13, generator=generator, special_embedding=embedding.embedding, num_inference_steps=1, return_dict=False)
        videos = pipe(p, negative_prompt="", num_videos_per_prompt=2, num_frames=13, generator=generator, special_embedding=embedding.embedding, num_inference_steps=50, dump_attention_maps=dump_attention_maps).frames

        print(f"num attn maps = {len(map_storage)}")
        # print(f"len = {len(attn_maps)}")
        # for j, attn_output in enumerate(attn_outputs):
        #     print(f"Attention output {j} shape: {attn_output.shape}")
        #     output_path = f"./step_{step}_seed_{seed}_p_{p_idx}_attn_output_{j}.mp4"
        #     print(f"Saving attention output {j} to {output_path}")
        #     export_to_video(attn_output, output_path, fps=8)
        for j, video in enumerate(videos):
            if dump_attention_maps:
                save_attn_map(map_storage[-1][j], video[0], step, seed, p_idx, p, j)
            output_path = f"./step_{step}_seed_{seed}_p_{p_idx}_{j}.mp4"
            print(f"Saving video {j} to {output_path}")
            export_to_video(video, output_path, fps=8)
            # exit(0)

# seed = 52
# step = 2000
# generator = torch.Generator(device=pipe.transformer.device).manual_seed(seed)
# videos = pipe("A * dog running in the snow", negative_prompt="", num_videos_per_prompt=2, num_frames=13, generator=generator).frames
# for j, video in enumerate(videos):
#     output_path = f"./ti_step_{step}_{seed}_seed_{j}.mp4"
#     print(f"Saving video {j} to {output_path}")
#     export_to_video(video, output_path, fps=8)
# video = pipe("A * dog at the beach", embedding.embedding, num_frames=13).frames[0]
# export_to_video(video, "output.mp4", fps=8)