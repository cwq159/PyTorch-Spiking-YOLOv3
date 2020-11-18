from spiking_utils.spike_dag import *
from spiking_utils.spike_layer import *

dag = None


def find_node_by_tensor(tensor):
    rst = [k for k, v in dag.nodes.items() if v is tensor]
    if len(rst) == 0:
        raise ValueError("cannot find tensor Size", tensor.size())
    elif len(rst) > 1:
        raise ValueError("More than one nodes save the tensor Size", tensor.size())
    return rst[0]


def find_op_by_out_node(node_name):
    for op_name, op in dag.ops.items():
        if node_name in op['out_nodes']:
            return op
    raise ValueError(f"cannot find node {node_name}")


view_count = 0


class WrappedTensor(torch.Tensor):
    def view(self, *size):
        global view_count
        view_count += 1
        op_name = f'view{view_count}'
        in_nodes = [find_node_by_tensor(self)]
        op = DAGViewOp(size)
        out = super().view(*size)
        out = WrappedTensor(out)
        out_nodes = [f'{op_name}_out{1}']
        dag.add_op(op_name, op, in_nodes, out_nodes)
        dag.add_node(out_nodes[0], out)
        return out


wrapped_functions = {}
relu_count = 0
conv2d_count = 0
conv_transpose2d_count = 0
batch_norm_count = 0
avg_pool2d_count = 0
linear_count = 0
concat_count = 0


def relu_wrapper(inp, inplace=False):
    global relu_count
    relu_count += 1
    op_name = f'relu{relu_count}'
    in_nodes = [find_node_by_tensor(inp)]
    op = SpikeReLU()
    out = wrapped_functions["relu"](inp, inplace=False)
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


def conv2d_wrapper(inp, weight, bias, stride, padding, dilation, groups):
    global conv2d_count
    conv2d_count += 1
    op_name = f'conv{conv2d_count}'
    in_nodes = [find_node_by_tensor(inp)]
    op = SpikeConv2d(inp.size(1), weight.size(0), weight.size()[2:], stride, padding, dilation, groups)
    op.weight = weight
    op.bias = bias
    out = wrapped_functions["conv2d"](inp, weight, bias, stride, padding, dilation, groups)
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


def conv_transpose2d_wrapper(inp, weight, bias, stride, padding, out_padding, groups, dilation):
    global conv_transpose2d_count
    conv_transpose2d_count += 1
    op_name = f'conv_transpose2d{conv_transpose2d_count}'
    in_nodes = [find_node_by_tensor(inp)]
    op = SpikeConvTranspose2d(inp.size(1), weight.size(0), weight.size()[2:], stride, padding, out_padding, groups,
                              bias is not None, dilation)
    op.weight = weight
    op.bias = bias
    out = wrapped_functions["conv_transpose2d"](inp, weight, bias, stride, padding, out_padding, groups, dilation)
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


def batch_norm_wrapper(inp, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    # fuse conv and bn
    global batch_norm_count
    batch_norm_count += 1
    in_nodes = [find_node_by_tensor(inp)]
    in_op = find_op_by_out_node(in_nodes[0])
    if not (isinstance(in_op['op'], SpikeConv2d) or isinstance(in_op['op'], SpikeConvTranspose2d)):
        raise ValueError(
            f"Conv2d/ConvTranspose2d is expected before BatchNorm, but {type(in_op['op'])} found. \n {in_op}")
    bn = nn.BatchNorm2d(weight.size(0), eps, True).to(inp.device)
    bn.eval()
    bn.weight.data = weight.data
    bn.bias.data = bias.data
    bn.running_mean.data[...] = running_mean
    bn.running_var.data[...] = running_var
    in_op['op'].bn = bn
    out = wrapped_functions["batch_norm"](inp, weight, bias, running_mean, running_var, training, momentum, eps,
                                          cudnn_enabled)
    out = WrappedTensor(out)
    dag.nodes[in_nodes[0]] = out
    return out


def avg_pool2d_wrapper(inp, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                       divisor_override=None):
    global avg_pool2d_count
    avg_pool2d_count += 1
    op_name = f'avg_pool2d{avg_pool2d_count}'
    in_nodes = [find_node_by_tensor(inp)]
    op = SpikeAvgPool2d(kernel_size, stride, padding)
    out = wrapped_functions["avg_pool2d"](inp, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                          divisor_override)
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


def linear_wrapper(inp, weight, bias=None):
    global linear_count
    linear_count += 1
    op_name = f'fc{linear_count}'
    in_nodes = [find_node_by_tensor(inp)]
    op = SpikeLinear(inp.size(1), weight.size(0))
    op.weight = weight
    op.bias = bias
    out = wrapped_functions["linear"](inp, weight, bias)
    out = WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


def concat_wrapper(tensors, dim=None):
    global concat_count
    concat_count += 1
    op_name = f"concat{concat_count}"
    in_nodes = [find_node_by_tensor(tensor) for tensor in tensors]
    op = DAGConcatOp(dim=dim)
    print(type(tensors[0]))
    out = wrapped_functions['concat'](tensors, dim)
    # out=WrappedTensor(out)
    out_nodes = [f'{op_name}_out{1}']
    dag.add_op(op_name, op, in_nodes, out_nodes)
    dag.add_node(out_nodes[0], out)
    return out


def wrap():
    raw = F.relu
    wrapped_functions["relu"] = raw
    F.relu = relu_wrapper

    raw = F.conv2d
    wrapped_functions["conv2d"] = raw
    F.conv2d = conv2d_wrapper

    raw = F.conv_transpose2d
    wrapped_functions["conv_transpose2d"] = raw
    F.conv_transpose2d = conv_transpose2d_wrapper

    raw = torch.batch_norm
    wrapped_functions["batch_norm"] = raw
    torch.batch_norm = batch_norm_wrapper

    raw = F.avg_pool2d
    wrapped_functions["avg_pool2d"] = raw
    F.avg_pool2d = avg_pool2d_wrapper

    raw = F.linear
    wrapped_functions["linear"] = raw
    F.linear = linear_wrapper

    raw = torch.cat
    wrapped_functions["concat"] = raw
    torch.cat = concat_wrapper


def unwrap():
    F.relu = wrapped_functions["relu"]
    F.conv2d = wrapped_functions["conv2d"]
    F.conv_transpose2d = wrapped_functions["conv_transpose2d"]
    torch.batch_norm = wrapped_functions['batch_norm']
    F.avg_pool2d = wrapped_functions["avg_pool2d"]
    F.linear = wrapped_functions["linear"]
    torch.cat = wrapped_functions['concat']


def parse_ann_model(model, inputs):
    global dag
    dag = SpikeDAGModule()
    model.eval()
    model.cpu()
    warpped_input = []
    for i, x in enumerate(inputs):
        inp = WrappedTensor(x.cpu())
        warpped_input.append(inp)
        name = f'dag_input{i}'
        dag.nodes[name] = inp
        dag.inputs_nodes.append(name)
    wrap()
    model(*warpped_input)
    unwrap()
    dag.clear_nodes()
    dag.outputs_nodes = dag.find_end_nodes()
    return dag
