import argparse

from torch.utils.data import DataLoader

from spiking_utils.snn_evaluate import *
from spiking_utils.snn_transformer import SNNTransformer
from utils.datasets import *
from utils.utils import *

# Hyperparameters
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])


class ListWrapper(nn.Module):
    """
    partial model(without route & conv[-1] & yolo layers) to transform
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
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='initial weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 640], help='[min_train, max-train, test]')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--statistics_iters', default=30, type=int, help='iterations for gather activation statistics')
    parser.add_argument('--timesteps', '-T', default=16, type=int)
    parser.add_argument('--reset_mode', default='subtraction', type=str, choices=['zero', 'subtraction'])
    parser.add_argument('--channel_wise', '-cw', action='store_true', help='transform in each channel')
    parser.add_argument('--save_file', default="yolov3-tiny-ours-snn", type=str,
                        help='the output location of the transferred weights')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
    print(opt)
    device = torch_utils.select_device(opt.device, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    cfg = opt.cfg
    data = opt.data
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

    # Image Sizes
    gs = 32  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
    img_size = imgsz_max  # initialize with max size

    # Configure run
    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Initialize model
    ann = Darknet(cfg).to(device)
    if opt.weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        ckpt = torch.load(weights, map_location=device)

        # load model
        try:
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if ann.state_dict()[k].numel() == v.numel()}
            ann.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

    # Partial net
    ann_to_transform = ListWrapper(ann.module_list)

    # # Test the partial_ann results
    # if opt.task == 'test':  # (default) test normally
    #     ann_results, maps = ann_evaluate(opt,
    #                                      data,
    #                                      ann,
    #                                      ann_to_transform,
    #                                      batch_size,
    #                                      imgsz_test,
    #                                      opt.conf_thres,
    #                                      opt.iou_thres,
    #                                      opt.save_json,
    #                                      opt.single_cls,
    #                                      opt.augment)

    # Transform
    transformer = SNNTransformer(opt, ann_to_transform, device)
    # calculate the statistics for parameter-normalization with train_dataloader
    transformer.inference_get_status(dataloader, opt.statistics_iters)
    snn = transformer.generate_snn()

    # Test the snn results
    if opt.task == 'test':  # (default) test normally
        snn_results, maps, firing_ratios = snn_evaluate(opt,
                                                        data,
                                                        ann,
                                                        snn,
                                                        opt.timesteps,
                                                        batch_size,
                                                        imgsz_test,
                                                        opt.conf_thres,
                                                        opt.iou_thres,
                                                        opt.save_json,
                                                        opt.single_cls,
                                                        opt.augment)

    # Save the SNN
    torch.save(snn, opt.save_file + '.pth')
    torch.save(snn.state_dict(), opt.save_file + '.weight')
    print("Save the SNN in {}".format(opt.save_file))

    # Save the snn info
    snn_info = {
        # 'ann_mAP': ann_results[2],
        'snn_mAP': snn_results[2],
        'mean_firing_ratio': float(firing_ratios.mean()),
        'firing_ratios': [float(_) for _ in firing_ratios],
    }
    with open(opt.save_file + '.json', 'w') as f:
        json.dump(snn_info, f)
