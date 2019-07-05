import os, os.path

DIR = 'base_dir/train_dir/forward'
print (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))