# RestNet Mini-Project 1

This repository is for NYU ECE-7123 deep learning Mini-Project1. Team members:

* Name: Shunyi Zhu  NetID:sz3719
* Name:Zhen Wang   NetID:zw2655
* Name: Haoze He     NetID: hh2537



### Run the Code

To run the code, you can run `project1.py --bs 128 --lr 0.1 --decay_step 40 --checkpoint "{}/resnet-18.log" --smooth`

* `bs`: batch size, default 128
* `lr`: learning rate, default 0.1
* `decay_ step`: decay step default 40
* `checkpoint`: file saving directory, default "{}/resnet-18.log"
* `smooth`: Label Smoothing Cross Entropy or Simple Cross Entropy, defalut Label Smoothing Cross Entropy



### Best Configuration and Architecture

To get the best result, you can run `python project1_model.py --bs 64 --lr 0.1 --decay_step 50`

* To load the final model, please refer to `resnet-18/project1_model.pt`.
* To see the best result, please refer to `./result/opti.txt` . The accuracy is higher than 93%, please refer to `final_model_plot.png`.
* To see figures of other experiments, please refer to `adam.png`,`batch_size.png`, `sgd_cross.png`, `decay_step.png`, and`learning_rate.png`.
* To see the parameters of the Resnet, please refer to`para.txt`.



