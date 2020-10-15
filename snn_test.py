from torch.utils.data import DataLoader

import spike_tensor
from spike_tensor import SpikeTensor
from utils.datasets import *
from utils.utils import *

import os
import torch
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ann_evaluate(ann, partial_ann, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    ann.eval()
    partial_ann.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            output1, output2 = partial_ann(imgs)  # two branches
            # post-processing: conv, yolo, nms
            output1 = ann.module_list[14](output1)
            output2 = ann.module_list[-2](output2)
            yolo_outputs = []
            x1, _ = ann.module_list[15][0](output1, img_dim=img_size)
            yolo_outputs.append(x1)
            x2, _ = ann.module_list[-1][0](output2, img_dim=img_size)
            yolo_outputs.append(x2)
            yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
            outputs = non_max_suppression(yolo_outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0], dtype=np.int)
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def snn_evaluate(ann, snn, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, timesteps):
    ann.eval()
    snn.eval()
    ann.to(device)
    snn.to(device)


    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    total_firing_ratios = []
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        replica_data = torch.cat([imgs for _ in range(timesteps)], 0)  # replica for input(first) layer
        data = SpikeTensor(replica_data, timesteps, scale_factor=1)

        with torch.no_grad():
            spike_tensor.firing_ratio_record = True
            output_snn1, output_snn2 = snn(data)  # two branches
            #if output_snn1.size(1) != 512:
            #    output_snn1, output_snn2 = output_snn2, output_snn1
            #assert output_snn1.size(1) == 512 and output_snn2.size(1) == 256
            #print('output_snn shape: ', output_snn1.size(), output_snn2.size())
            spike_tensor.firing_ratio_record = False
            output_ann1 = output_snn1.to_float()  # spike to real-value
            output_ann2 = output_snn2.to_float()
            #print('output: ', output_ann1[0], output_ann2[0])
            #print('output_ann shape: ', output_ann1.size(), output_ann2.size())
            # post-processing: conv, yolo, nms
            output_ann1 = ann.module_list[14](output_ann1)
            output_ann2 = ann.module_list[-2](output_ann2)
            yolo_outputs = []
            x1, _ = ann.module_list[15][0](output_ann1, img_dim=img_size)
            yolo_outputs.append(x1)
            x2, _ = ann.module_list[-1][0](output_ann2, img_dim=img_size)
            yolo_outputs.append(x2)
            yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
            outputs = non_max_suppression(yolo_outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

        total_firing_ratios.append([_.mean().item() for _ in spike_tensor.firing_ratios])
        spike_tensor.firing_ratios = []

    total_firing_ratios = np.mean(total_firing_ratios, 0)
    mean_firing_ratio = total_firing_ratios.mean()
    print(f"Mean Firing ratios {mean_firing_ratio}, Firing ratios: {total_firing_ratios}")

    for layer in snn.modules():
        if hasattr(layer, 'mem_potential'):
            layer.mem_potential = None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class, total_firing_ratios
