import torch
from finetrainers.models.wan.inv_specification import WanPipelineInv
from diffusers.utils import export_to_video
import safetensors

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
    
pipe = WanPipelineInv.from_pretrained(
    "/ocean/model/Wan-AI/Wan2.1-T2V-14B-Diffusers", torch_dtype=torch.bfloat16
).to("cuda")

# Load the embedding from a file
embedding_tensor = torch.Tensor(1, 1, 4096)
embedding = EmbeddingWrap(embedding_tensor)
safetensors.torch.load_model(embedding, "./demo/output-class/finetrainers_step_2000/model.safetensors")
# pipe.load_lora_weights("/home/jrguo/finetrainers/demo2/output/lora_weights/000500", adapter_name="wan-lora")
# pipe.set_adapters(["wan-lora"], [0.75])

seed = 52
step = 3000
generator = torch.Generator(device=pipe.transformer.device).manual_seed(seed)
videos = pipe("A * dog running in the snow", negative_prompt="", num_videos_per_prompt=2, num_frames=13, generator=generator).frames
for j, video in enumerate(videos):
    output_path = f"./ti_step_{step}_{seed}_seed_{j}.mp4"
    print(f"Saving video {j} to {output_path}")
    export_to_video(video, output_path, fps=8)
# video = pipe("A * dog at the beach", embedding.embedding, num_frames=13).frames[0]
# export_to_video(video, "output.mp4", fps=8)