# README

type python p1.py --bs 128 --lr 0.1 --decay_step 40 --checkpoint "{}/resnet-18.log" --smooth

bs: batch size, default 128

lr: learning rate, default 0.1

decay_ step: decay step default 40

checkpoint: file saving directory, default "{}/resnet-18.log"

smooth: Label Smoothing Cross Entropy or Simple Cross Entropy, defalut Label Smoothing Cross Entropy