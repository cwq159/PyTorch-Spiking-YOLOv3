# PyTorch-Spiking-YOLOv3
A minimal PyTorch implementation of Spiking-YOLOv3, based on the minimal PyTorch implementation of PyTorch-YOLOv3([eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)), with support for spiking-yolov3-tiny at present.

## Introduction
There is a little difference between spiking-yolov3-tiny and yolov3-tiny, please refer to *.cfg for details.
### Transformation of some operators
+ 'maxpool'->'avgpool'
+ 'upsample'->'transposed_convolutional'
+ 'leaky_relu'->'relu'

## Usage
Please refer to [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) for the basic usage of PyTorch-YOLOv3 for training, inference and evaluation.
### Train
```
python3 train.py --model_def config/spiking-yolov3-tiny.cfg 
```
### Transform
```
python3 ann_to_snn.py --model_def config/spiking-yolov3-tiny.cfg --weights_path checkpoints/yolov3_ckpt_100.pth --channel_wise
```

## References
### Articles
+ [Theory and Tools for the Conversion of Analog to Spiking Convolutional Neural Networks](https://arxiv.org/abs/1612.04052)
+ [Spiking-YOLO: Spiking Neural Network for Energy-Efficient Object Detection](https://arxiv.org/abs/1903.06530)
### GitHub
+ [NeuromorphicProcessorProject/snn_toolbox](https://github.com/NeuromorphicProcessorProject/snn_toolbox)
+ [hahnyuan/ANN2SNN](http://git.wildz.cn/hahnyuan/ANN2SNN)
