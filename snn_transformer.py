import torch
import torch.nn.functional as F

from spike_layer import *
import ann_parser


def is_layer_weighted_spike(layer):
    return isinstance(layer, SpikeConv2d) or isinstance(layer, SpikeLinear) or isinstance(layer, SpikeConvTranspose2d)


class DataStatus():
    def __init__(self, max_num=1e7, channel_wise=True):
        self.pool = []
        self.num = 0
        self.max_num = max_num
        self.channel_wise = channel_wise

    def append(self, data):
        if self.channel_wise:
            b, c = data.size()[:2]
            self.pool.append(data.transpose(0, 1).contiguous().view(c, -1))
            self.num += self.pool[-1].size()[0] * self.pool[-1].size()[1]
        else:
            self.pool.append(data.view(-1))
            self.num += self.pool[-1].size()[0]
        if self.num > self.max_num:
            self.random_shrink()

    def random_shrink(self):
        if self.channel_wise:
            tensor = torch.cat(self.pool, 1)
            c, n = tensor.size()[:2]
            tensor = tensor[:, torch.randint(n, size=[int(n // 2)])]
        else:
            tensor = torch.cat(self.pool, 0)
            tensor = tensor[torch.randint(
                len(tensor), size=[int(self.max_num // 2)])]
        self.pool.clear()
        self.pool.append(tensor)

    def max(self, fraction=1, relu=True, max_num=1e6):
        if self.channel_wise:
            tensor = torch.cat(self.pool, 1)  # shape [n_channels, n]
        else:
            tensor = torch.cat(self.pool, 0)  # shape [n]
        if relu:
            tensor = F.relu(tensor)
        if self.channel_wise:
            tensor_sort = tensor.sort(1)[0]
            return tensor_sort[:, int(fraction * tensor_sort.size(1))]
        else:
            tensor_sort = tensor.sort()[0]
            return tensor_sort[int(fraction * tensor_sort.size(0))]


class SNNTransformer():
    def __init__(self, args, net, device):
        self.original_net = net
        self.timesteps = args.timesteps
        self.device = device
        self.snn_dag = None
        self.ann_snn_layer_mapping = {}
        self.reset_mode = args.reset_mode
        self.layer2name = {}
        self.input_status = {}
        self.output_status = {}
        self.input_generator = {}
        self.channel_wise = args.channel_wise

    def init_dag(self, inputs):
        self.snn_dag = ann_parser.parse_ann_model(self.original_net, inputs)
        self.snn_dag.to(self.device)
        # trace spike layers
        for layer_name, layer in self.snn_dag.named_modules():
            self.layer2name[layer] = layer_name
            if is_layer_weighted_spike(layer):
                self.input_status[layer_name] = DataStatus(
                    channel_wise=self.channel_wise)
                self.output_status[layer_name] = DataStatus(
                    channel_wise=self.channel_wise)

                def forward_hook(m, inputs, outputs):
                    self.input_status[self.layer2name[m]].append(
                        inputs[0].detach().cpu())
                    self.output_status[self.layer2name[m]].append(
                        outputs.detach().cpu())

                layer.register_forward_hook(forward_hook)

    def inference_get_status(self, train_loader, num_iters):
        for batch_i, (_, imgs, targets) in enumerate(train_loader):
            if batch_i > num_iters:
                break
            data = imgs.to(self.device)
            if self.snn_dag is None:
                self.init_dag([data])
            out = self.snn_dag(data)
        # freeze hook of spike layers
        for layer_name, layer in self.snn_dag.named_modules():
            layer._forward_hooks.clear()

    def fuse_bn(self):
        for layer_name, layer in self.snn_dag.named_modules():
            if isinstance(layer, SpikeConv2d) and layer.bn is not None:
                layer.weight.data[...] = layer.weight * (
                            layer.bn.weight / torch.sqrt(layer.bn.running_var + layer.bn.eps)).view(-1, 1, 1, 1)
                if layer.bias is not None:
                    layer.bias.data[...] = (layer.bias - layer.bn.running_mean) * layer.bn.weight / torch.sqrt(
                        layer.bn.running_var + layer.bn.eps) + layer.bn.bias
                else:
                    bias = (-layer.bn.running_mean) * layer.bn.weight / torch.sqrt(
                        layer.bn.running_var + layer.bn.eps) + layer.bn.bias
                    bias = nn.Parameter(bias)
                    layer._parameters['bias'] = bias
                    layer.bias = bias
                layer.bn = None
                print(f"Fuse the weights in {layer_name}")

    def gen_weight(self, layer, max_in, max_out):
        weight = layer.weight
        if len(weight.size()) == 4:
            scale_snn = max_in.view(1, -1, 1, 1) / max_out.view(-1, 1, 1, 1)
        elif len(weight.size()) == 2:
            scale_snn = max_in.view(1, -1) / max_out.view(-1, 1)
        else:
            raise NotImplementedError
        return weight.data * scale_snn

    def gen_bias(self, layer, max_out):
        return layer.bias.data / max_out

    def gen_Vthr(self, layer):
        return 1

    def generate_snn(self):
        self.fuse_bn()
        for layer_i, (layer_name, layer) in enumerate(self.snn_dag.named_modules()):
            if is_layer_weighted_spike(layer):
                print(f"processing layer {layer_name}")
                # TODO: supporting specify the first layer for multi input branch network
                input_status = self.input_status[layer_name]
                output_status = self.output_status[layer_name]
                max_in = input_status.max(fraction=0.99).to(self.device)
                max_out = output_status.max(fraction=0.99).to(self.device)
                if layer_i == 0:
                    layer.weight.data[...] = self.gen_weight(
                        layer, torch.ones(1).to(self.device), max_out)
                else:
                    layer.weight.data[...] = self.gen_weight(
                        layer, max_in, max_out)
                layer.Vthr[...] = self.gen_Vthr(layer)
                layer.out_scales.data[...] = max_out
                if layer.bias is not None:
                    layer.bias.data[...] = self.gen_bias(layer, max_out)
                    layer.leakage = layer.bias.data
                print(f"set {layer_name}: Vthr {layer.Vthr}")
        # unwrap the layers
        for layer in self.snn_dag.modules():
            if is_layer_weighted_spike(layer):
                layer._forward_hooks.clear()
        for m in self.snn_dag.modules():
            m.reset_mode = self.reset_mode
        print(f"Transfer ANN to SNN Finished")
        return self.snn_dag
