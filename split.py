# Load a video and evenly divide it in temporal dimension
import torch
import argparse
import cv2
from tqdm import tqdm
import os
from diffusers.utils import export_to_video
from diffusers.utils.loading_utils import load_video

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Split a video tensor into equal parts.")
    parser.add_argument("--num_splits", type=int, default=3, help="Number of splits to divide the video into.")
    parser.add_argument("--input", type=str, default='./demo/data/0.mp4', help="Path to the video.")
    parser.add_argument("--output", type=str, default=None, help="Path to the output directory.")
    args = parser.parse_args()

    # Load input video (mp4)
    print("Loading video from:", args.input)
    frames = load_video(args.input)

    T = len(frames)
    t = (T-50) // args.num_splits

    # Save the splited videos
    for i in tqdm(range(args.num_splits)):
        output_path = os.path.join(os.path.dirname(args.input), f"split_video_part_{i}.mp4")

        if args.output:
            output_path = os.path.join(args.output, f"split_video_part_{i}.mp4")
            os.makedirs(args.output, exist_ok=True)

        print(f"Saving split video part {i} to: {output_path}")
        start = i * t
        end = (i + 1) * t if i < args.num_splits - 1 else T
        split_frames = frames[start:end]
        export_to_video(split_frames, output_path, fps=30)

        # Copy caption file
        caption_file = os.path.splitext(args.input)[0] + ".txt"
        if os.path.exists(caption_file):
            with open(caption_file, 'r') as f:
                caption = f.read()
            with open(os.path.splitext(output_path)[0] + ".txt", 'w') as f:
                f.write(caption)
            print(f"Copied caption to: {os.path.splitext(output_path)[0] + '.txt'}")

    print("Video splitting completed. Sub-videos length:", t, "frames each.")
