import argparse
import json
from torch.utils.data import DataLoader

from models import *
from snn_test import snn_evaluate
from snn_test import ann_evaluate
from snn_transformer import SNNTransformer
from utils.datasets import *
from utils.utils import *

torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


class ListWrapper(nn.Module):
    """
    partial model(without route & yolo layers) to transform
    """
    def __init__(self, modulelist):
        super().__init__()
        self.list = modulelist

    def forward(self, x):
        for i in range(9):
            x = self.list[i](x)
        x1 = x  # route1
        for i in range(9, 13):
            x = self.list[i](x)
        x2 = x  # route2
        y1 = self.list[13](x)  # branch1
        c = self.list[17](x2)
        c = self.list[18](c)
        x = torch.cat((c, x1), 1)
        y2 = self.list[20](x)  # branch2
        return y1, y2


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument('--statistics_iters', default=30, type=int, help='iterations for gather activation statistics')
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument('--timesteps', '-T', default=16, type=int)
    parser.add_argument('--reset_mode', default='subtraction', type=str, choices=['zero', 'subtraction'])
    parser.add_argument('--channel_wise', '-cw', action='store_true', help='transform in each channel')
    parser.add_argument('--save_file', default="./out_snn.pth", type=str,
                        help='the output location of the transferred weights')

    args = parser.parse_args()
    args.activation_bitwidth = np.log2(args.timesteps)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Preparing the dataset
    data_config = parse_data_config(args.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    train_dataset = ListDataset(train_path, augment=True, multiscale=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    # Build Model
    ann = Darknet(args.model_def).to(device)
    if args.weights_path.endswith(".weights"):
        # Load darknet weights
        ann.load_darknet_weights(args.weights_path)
    else:
        # Load checkpoint weights
        ann.load_state_dict(torch.load(args.weights_path, map_location=device))
    ann_to_transform = ListWrapper(ann.module_list)

    # Test the results
    print("Compute ann_mAP...")

    precision, recall, ann_AP, f1, ap_class = ann_evaluate(
        ann,
        ann_to_transform,
        path=valid_path,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        img_size=args.img_size,
        batch_size=8,
    )

    print("ANN Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {ann_AP[i]}")

    print(f"ann_mAP: {ann_AP.mean()}")

    # Transform
    transformer = SNNTransformer(args, ann_to_transform, device)
    # calculate the statistics for parameter-normalization
    transformer.inference_get_status(train_loader, args.statistics_iters)
    snn = transformer.generate_snn()

    # Test the results
    print("Compute snn_mAP...")

    precision, recall, snn_AP, f1, ap_class, firing_ratios = snn_evaluate(
        ann,
        snn,
        path=valid_path,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        img_size=args.img_size,
        batch_size=8,
        timesteps=args.timesteps
    )

    print("SNN Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - snn_AP: {snn_AP[i]}")

    print(f"snn_mAP: {snn_AP.mean()}")

    # Save the SNN
    torch.save(snn, args.save_file)
    torch.save(snn.state_dict(), args.save_file + '.weight')
    print("Save the SNN in {}".format(args.save_file))

    # Save the snn info
    snn_info = {
        'ann_mAP': float(ann_AP.mean()),
        'snn_mAP': float(snn_AP.mean()),
        'mean_firing_ratio': float(firing_ratios.mean()),
        'firing_ratios': [float(_) for _ in firing_ratios],
    }
    with open(args.save_file + '.json', 'w') as f:
        json.dump(snn_info, f)
