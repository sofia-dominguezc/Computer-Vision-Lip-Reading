import argparse
import math
import os
import warnings

from data.data_module import AVSRDataLoader
from tqdm import tqdm
from transforms import TextTransform
from utils import save_vid_txt

warnings.filterwarnings("ignore")

# Argument Parsing
parser = argparse.ArgumentParser(description="WildVSR Preprocessing")
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
    "--subset",
    type=str,
    required=True,
    help="Subset of dataset",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Name of dataset",
)
parser.add_argument(
    "--gpu_type",
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
    "--combine-av",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Merges the audio and video components to a media file.",
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

args.data_dir = os.path.normpath(args.data_dir)
vid_dataloader = AVSRDataLoader(
    modality="video", detector=args.detector, convert_gray=False, gpu_type=args.gpu_type
)

seg_vid_len = seg_duration * 25

label_filename = os.path.join(
    args.root_dir,
    "labels",
    f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s.csv"
    if args.groups <= 1
    else f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s.{args.groups}.{args.job_index}.csv",
)
os.makedirs(os.path.dirname(label_filename), exist_ok=True)
print(f"Directory {os.path.dirname(label_filename)} created")

f = open(label_filename, "w")

dst_vid_dir = os.path.join(
    args.root_dir, dataset, dataset + f"_video_seg{seg_duration}s"
)
dst_txt_dir = os.path.join(
    args.root_dir, dataset, dataset + f"_text_seg{seg_duration}s"
)

import json
with open(os.path.join(args.data_dir, "labels.json"), "r") as file:
    data = json.load(file)
    filenames = [
        os.path.join(args.data_dir, "videos", k) for k in data
    ]

unit = math.ceil(len(filenames) * 1.0 / args.groups)
filenames = filenames[args.job_index * unit : (args.job_index + 1) * unit]

for data_filename in tqdm(filenames):
    video_data = vid_dataloader.load_data(data_filename, landmarks=None, transform=False)

    dst_vid_filename = (
        f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.mp4"
    )
    dst_txt_filename = (
        f"{data_filename.replace(args.data_dir, dst_txt_dir)[:-4]}.txt"
    )
    trim_vid_data = video_data

    content = data[os.path.basename(data_filename)]

    if trim_vid_data is None:
        continue
    video_length = len(trim_vid_data)
    if video_length == 0:
        continue
    save_vid_txt(
        dst_vid_filename,
        dst_txt_filename,
        trim_vid_data,
        content,
        video_fps=25,
    )

    basename = os.path.relpath(
        dst_vid_filename, start=os.path.join(args.root_dir, dataset)
    )
    token_id_str = " ".join(
        map(str, [_.item() for _ in text_transform.tokenize(content)])
    )
    f.write(
        "{}\n".format(
            f"{dataset},{basename},{trim_vid_data.shape[0]},{token_id_str}"
        )
    )
    
f.close()
