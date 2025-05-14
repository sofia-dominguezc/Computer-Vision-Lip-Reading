import argparse
import os
import warnings

from data.data_module import AVSRDataLoader
from transforms import TextTransform
from utils import save2vid

warnings.filterwarnings("ignore")

# Argument Parsing
parser = argparse.ArgumentParser(description="TCD-TIMIT Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    required=True,
    help="Directory of original dataset",
)
parser.add_argument(
    "--detector",
    type=str,
    default="mediapipe",
    help="Type of face detector. (Default: mediapipe)",
)
parser.add_argument(
    "--root-dir",
    type=str,
    required=True,
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Name of dataset",
)
parser.add_argument(
    "--gpu-type",
    type=str,
    default="cuda",
    help="GPU type, either mps or cuda. (Default: cuda)",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=16,
    help="Max duration (second) for each segment, (Default: 16)",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel.",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing).",
)
args = parser.parse_args()

seg_duration = args.seg_duration
dataset = args.dataset
text_transform = TextTransform()

# Load Data
args.data_dir = os.path.normpath(args.data_dir)

vid_dataloader = AVSRDataLoader(
    modality="video", detector=args.detector, convert_gray=False, gpu_type=args.gpu_type
)

seg_vid_len = seg_duration * 25

processed_labels_path = os.path.join(args.root_dir, "labels")
processed_videos_path = os.path.join(args.root_dir, f"{dataset}_videos")

os.makedirs(processed_labels_path, exist_ok=True)
os.makedirs(processed_videos_path, exist_ok=True)

transform = TextTransform()

with open(os.path.join(processed_labels_path, "labels.txt"), "a") as f:
    for dirpath, dirnames, filenames in os.walk(args.data_dir):
            if "lipspeakers" in dirpath:
                role = "lipspeaker"
            elif "volunteers" in dirpath:
                role = "volunteer"
            for num in range(20, 0, -1):
                if str(num) in dirpath:
                    number = num
                    break
            video_files = [
                file for file in filenames if file.endswith(".mp4")
            ]
            for video_file in video_files:
                video_path = os.path.join(dirpath, video_file)
                try:
                    #video_data = vid_dataloader.load_data(video_path, landmarks=None)
                    #save2vid(os.path.join(processed_videos_path, f"{role}-{number}-{video_file}"), video_data, frames_per_second=25)
                    if role == "volunteer":
                        label_file = video_file.upper().replace(".MP4", ".txt")
                    else:
                        label_file = video_file.replace(".mp4", ".txt")
                    label_path = os.path.join(dirpath, label_file)
                    with open(label_path, "r") as ff:
                        if role == "lipspeaker":
                            label = ff.readline().strip()[:-1]
                        else:
                            label = ' '.join(line.strip().split()[-1] for line in ff.readlines())
                        tokens = transform.tokenize(label.upper()).detach().cpu().tolist()
                        f.write(f"{dataset}_videos,{f'{role}-{number}-{video_file}'},{str(len(tokens))},{' '.join(map(str, tokens))}\n")
                except AssertionError:
                    continue
