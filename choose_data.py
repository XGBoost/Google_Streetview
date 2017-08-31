import os
import random

label1_path='/home/ying/data/google_streetview_train_test1/label.txt'
file=open(label1_path)
lines=file.readlines()  # line's type is a list
random.seed(1)  # set the random seed, and every seed generate the same random result

slice=random.sample(lines,200000)
tmp=random.shuffle(slice)  # choose 20w data randomly, and shuffle it
imgs=[]
root="/home/ying/data/google_streetview_train_test1"
train_label=slice[:150000]
test_label=slice[150000:200000]
c=0
for line in train_label:
    cls = line.split()
    fn = cls.pop(0)
    if os.path.isfile(os.path.join(root, fn)):
        imgs.append((fn, tuple([float(v) for v in cls[:len(cls)-1]])))
        # images is the list,and the content is the tuple, every image corresponds to a label
        # despite the label's dimension
        # we can use the append way to append the element for list
    c = c + 1



