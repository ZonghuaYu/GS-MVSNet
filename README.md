# GS-MVSNet

## Training
Download DTU dataset for train: [Baiduyun](https://pan.baidu.com/s/1PficbSLidkwedWqiegKM7A)  code：om7j

 Our model is trained on DTU dataset. Save in pth/model_16.pth 
 
 For training, Changing the directory of training dataset in the config.py("root_dir", "train_root_dir").  
 
Then, run 

    python traindtu.py 

## Testing

### DTU dataset

Download DTU dataset for test: [Baiduyun](https://pan.baidu.com/s/1Vy3LR7H1wUS_3m48tjF3wA)  code:sms1

1.set test dataset path in config.py("eval_root_dir").Then run

    python evaldtu.py -p pth/model_16.pth

### Tanks and Temples dataset

Download Tanks and Temples dataset for test: [Baiduyun](https://pan.baidu.com/s/1qsOgjbFEHgdRw89SEGg5ug)  code:g2a7

1.set test dataset path in config.py("datasetpath").Then run

    python evaltanks.py -p pth/model_16.pth
	
### fusion

2.install fusibile https://github.com/kysucix/fusibile

3.set fusibile location in tools/fusion/conf.py

4.set other paras in tools/fusion/conf.py

5.run

    python fusion.py -cfmgd -t dtu
    
    python fusion.py -cfmgd -t tanks
   
## Acknowledgements

Thanks to Yao Yao for opening source of his excellent work [MVSNet](https://github.com/YoYo000/MVSNet). Thanks to Xiaoyang Guo for opening source of his PyTorch implementation of MVSNet [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch).
