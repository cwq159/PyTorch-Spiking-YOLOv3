# PyTorch-Spiking-YOLOv3
A minimal PyTorch implementation of Spiking-YOLOv3, based on the minimal PyTorch implementation of YOLOv3([eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)), with support for Spiking-YOLOv3-Tiny at present. The whole Spiking-YOLOv3 will be supported soon.

## Introduction
For spiking implementation, some operators in YOLOv3-Tiny have been converted equivalently. Please refer to [yolov3-tiny-ours.cfg](/config/yolov3-tiny-ours.cfg) for details.
### Conversion of some operators
+ 'maxpool(stride=2)'->'convolutional(stride=2)'
+ 'maxpool(stride=1)'->'none'
+ 'upsample'->'transposed_convolutional'
+ 'leaky_relu'->'relu'
+ 'batch_normalization'->'fuse_conv_and_bn'

## Usage
Please refer to [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) for the basic usage of PyTorch-YOLOv3 for training, evaluation and inference. The main advantage of PyTorch-Spiking-YOLOv3 is the transformation from ANN to SNN.
### Train
```
$ python3 train.py
```
After training, please rename your checkpoint file and move it to the /weights folder.
```
$ cd checkpoints
$ mv yolov3-tiny-ours_ckpt_99.pth ../weights/yolov3-tiny-ours_best.pth
```
### Test
```
$ python3 test.py
```
### Detect
```
$ python3 detect.py
```
### Transform
```
$ python3 ann_to_snn.py
```
For higher accuracy(mAP), you can try to adjust some hyperparameters.

*Trick: the larger timesteps, the higher accuracy.*

## Results
Here we show the results(mAP) of COCO2014 which is commonly used in object detection，and two custom datasets UAV/UAVCUT.
|  dataset  |  yolov3  |  yolov3-tiny  |  yolov3-tiny-ours  |  yolov3-tiny-ours-snn  |
|  ----  |  ----  |  ----  |  ----  |  ----  |
|  UAVCUT  |  99.84%  |  99.86%  |  **99.80%**  |  **99.60%**  |
|  UAV  |  80.21%  |  90.81%  |  **89.05%**  |  **87.02%**  |
|  COCO2014  |  54.93%  |  30.87%  |  **13.30%**  |  **13.82%**  |
From the results, we can conclude that: 
1) for simple custom datasets, converting some operators is equivalent to the original YOLOv3-Tiny; 
2) for complete dataset like COCO2014, the accuracy of converting some operators is lower than the original YOLOv3-Tiny;
3) regardless of datasets, our method of transformation from ANN to SNN can be nearly lossless.

![avatar](/assets/uavcut.png)

![avatar](/assets/uav.png)

![avatar](/assets/dog.png)

## References
### Articles
+ [Theory and Tools for the Conversion of Analog to Spiking Convolutional Neural Networks](https://arxiv.org/abs/1612.04052)
+ [Spiking-YOLO: Spiking Neural Network for Energy-Efficient Object Detection](https://arxiv.org/abs/1903.06530)
### GitHub
+ [NeuromorphicProcessorProject/snn_toolbox](https://github.com/NeuromorphicProcessorProject/snn_toolbox)
+ [hahnyuan/ANN2SNN](http://git.wildz.cn/hahnyuan/ANN2SNN)
