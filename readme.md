# ADENet
This repository is for ADENet introduced in the following paper: J. Xiong, Y. Zhou, P. Zhang, L. Xie, W. Huang and Y. Zha, "[Look&Listen: Multi-Modal Correlation Learning for Active Speaker Detection and Speech Enhancement](https://ieeexplore.ieee.org/document/9858007)", (IEEE Transactions on Multimedia, 2022)


Project link:[ADENet](https://overcautious.github.io/ADENet/)
![ADENet.png](utils/ADENet.png)


### Dependencies

Start from building the environment
```
conda env create -f env.yaml
```

***

## Training 
### Data preparation
#### 1.AVA dataset
The following script can be used to download and prepare the AVA dataset for training.

```
python trainADENet.py --dataPathAVA AVADataPath --download 
```

`AVADataPath` is the folder you want to save the AVA dataset and its preprocessing outputs, the details can be found in `/utils/tools.py`. Please read them carefully.

#### 2.MUSAN noise dataset
1. Download the MUSAN dataset [openslr](https://www.openslr.org/17/)
2. Using following script can clip noise audio, generate training set, validation set, test set
```
python generate_speech_noise 
```

### Training

Then you can train ADENet in AVA end-to-end by using:
```
python trainADENet.py --dataPathAVA AVADataPath ----dataPathMUSAN MUSANDataPath --savePath savePath
```

Using parameter `--isDown, --isDown` to control  cross-modal circulant fusion
***

### Citation

Please cite the following if our paper or code is helpful to your research.
```
@ARTICLE{9858007,
  author={Xiong, Junwen and Zhou, Yu and Zhang, Peng and Xie, Lei and Huang, Wei and Zha, Yufei},
  journal={IEEE Transactions on Multimedia}, 
  title={Look&amp;listen: Multi-Modal Correlation Learning for Active Speaker Detection and Speech Enhancement}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TMM.2022.3199109}}
```

This is my first open-source work, please let me know if I can future improve in this repositories or there is anything wrong in our work. Thanks for your support!

