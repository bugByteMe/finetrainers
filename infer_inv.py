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
safetensors.torch.load_model(embedding, "./demo/output/finetrainers_step_2000/model.safetensors")
# pipe.load_lora_weights("/home/jrguo/finetrainers/demo2/output/lora_weights/000500", adapter_name="wan-lora")
# pipe.set_adapters(["wan-lora"], [0.75])

video = pipe("A car in a parking lot", embedding.embedding, num_frames=13).frames[0]
export_to_video(video, "output.mp4", fps=8)