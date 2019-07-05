import glob
import os
import random
for f in glob.glob("./base_dir/train_dir/*/*.jpg"):
    c = random.randint(1,11)
    if c == 6:
        os.rename(f, f.replace("train_dir", "val_dir"))