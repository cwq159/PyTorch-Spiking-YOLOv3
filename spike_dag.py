from collections import OrderedDict
import torch
import torch.nn as nn

from spike_tensor import SpikeTensor


class DAGViewOp:
    """
    Wrapper of the x.view operation
    """

    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, x):
        return x.view(*self.sizes)


class DAGConcatOp:
    """
    Wrapper of torch.cat() operation
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, *xs):
        if isinstance(xs[0], SpikeTensor):
            out = torch.cat([_.data for _ in xs], dim=self.dim)
            out = SpikeTensor(out, xs[0].timesteps, xs[0].scale_factor)
        else:
            out = torch.cat(xs, dim=self.dim)
        return out


class SpikeDAGModule(nn.Module):
    """
    This is a module contains `nodes` dict and `ops` dict.
    It can do inference under the topological ordering of the ops given the input nodes.
    - Attributes
        - `nodes` OrderedDict, in which the key is the name of the input (`<op_name>_out_<id>`)
        - `ops` OrderedDict, in which the key is the name of the op (`<op_type>_<id>`)
            and the value is a dict `{'op':op, 'in_nodes':tuple, 'out_nodes':tuple}`
        - `inputs_nodes` tuple, specifying the input nodes for forward
    - forward
        - copy the inputs to nodes in `inputs_nodes`
        - for `op` in `ops`:
            - get inputs from `nodes`
            - forward `op`
            - save outputs to `nodes`
    """

    def __init__(self):
        super().__init__()
        self.nodes = OrderedDict()
        self.ops = OrderedDict()
        self.inputs_nodes = []
        self.outputs_nodes = []

    def add_op(self, name, op, in_nodes, out_nodes):
        assert name not in self.ops, AssertionError(f"Error: op {name} already in dag.ops")
        assert len(in_nodes) and len(out_nodes)
        self.ops[name] = {'op': op, 'in_nodes': in_nodes, 'out_nodes': out_nodes}
        if isinstance(op, nn.Module):
            self.add_module(name, op)
        print(f"add node {name}: {in_nodes}->{out_nodes}")

    def add_node(self, name, tensor=None):
        assert name not in self.nodes, AssertionError(f"Error: node {name} already in dag.nodes")
        self.nodes[name] = tensor

    def clear_nodes(self):
        for key in self.nodes:
            self.nodes[key] = None

    def find_end_nodes(self):
        in_nodes = []
        for _, op in self.ops.items():
            in_nodes.extend(op['in_nodes'])
        end_nodes = set(self.nodes.keys()) - set(in_nodes)
        print(end_nodes)
        return tuple(end_nodes)

    def forward(self, *inputs):
        assert len(self.inputs_nodes) == len(inputs)
        assert len(self.outputs_nodes)
        for inp, name in zip(inputs, self.inputs_nodes):
            self.nodes[name] = inp
        for op_name, op in self.ops.items():
            op_inputs = tuple(self.nodes[_] for _ in op['in_nodes'])
            op_outputs = op['op'](*op_inputs)
            if isinstance(op_outputs, torch.Tensor) or isinstance(op_outputs, SpikeTensor):
                op_outputs = [op_outputs]
            for node_name, op_output in zip(op['out_nodes'], op_outputs):
                self.nodes[node_name] = op_output
        outputs = tuple(self.nodes[_] for _ in self.outputs_nodes)
        return outputs if len(outputs) > 1 else outputs[0]
