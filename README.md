# GS-MVSNet

## Training

 Our model is trained on [DTU dataset](https://pan.baidu.com/s/1PficbSLidkwedWqiegKM7A)(extraction code：om7j). Save in pth/model_16.pth 
 
 For training, Changing the directory of training dataset in the config.py("root_dir", "train_root_dir").  
 
Then, run 

    python traindtu.py 

## Testing

### [DTU dataset](https://pan.baidu.com/s/1Vy3LR7H1wUS_3m48tjF3wA )(extraction code:sms1)

1.set test dataset path in config.py("eval_root_dir").Then run

    python evaldtu.py -p pth/model_16.pth

### [Tanks and Temples dataset](https://pan.baidu.com/s/1qsOgjbFEHgdRw89SEGg5ug )(extraction code:g2a7)

1.set test dataset path in config.py("datasetpath").Then run

    python evaltanks.py -p pth/model_16.pth
	
### fusion

2.install fusibile https://github.com/kysucix/fusibile

3.set fusibile location in tools/fusion/conf.py

4.set other paras in tools/fusion/conf.py

5.run

    python fusion.py -cfmgd -t dtu
    
    python fusion.py -cfmgd -t tanks
