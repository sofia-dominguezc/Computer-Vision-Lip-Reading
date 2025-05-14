import numpy as np

from datamodule import transforms

transform = transforms.TextTransform()

def detokenize_labels():
    with open("trainset/labels.txt", "r") as f:
        with open("trainset/labels_detok.txt", "w") as ff:
            lines = f.readlines()
            for line in lines:
                dirname, filename, length, tokens = line.split(',')
                token_ids = np.array([int(token) for token in tokens.split()])
                text = transform.post_process(token_ids)
                ff.write(text + '\n') 

def change_labels(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        newlines = []
        for line in lines:
            dataset, filename, length, label = line.split(',')
            newline = ["trainset-videos", filename, length, label]
            newlines.append(','.join(newline))
    with open(filepath, "w") as f:
        for newline in newlines:
            if newline[-1] == "\n":
                f.write(newline)
            else:
                f.write(newline + "\n")

if __name__ == "__main__":
    change_labels("trainset/labels.txt")
