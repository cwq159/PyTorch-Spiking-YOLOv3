import torch

firing_ratio_record = False
firing_ratios = []


class SpikeTensor:
    def __init__(self, data, timesteps, scale_factor):
        """
        data shape: [t*batch, channel, height, width]
        """
        self.data = data
        self.timesteps = timesteps
        self.b = self.data.size(0) // timesteps
        self.chw = self.data.size()[1:]
        self.scale_factor = torch.ones([*self.data.size()[1:]]).to(data.device)
        if isinstance(scale_factor, torch.Tensor):
            dim = scale_factor.dim()
            if dim == 1:
                self.scale_factor *= scale_factor.view(-1, *([1] * (len(self.chw) - 1)))
            else:
                self.scale_factor *= scale_factor
        else:
            self.scale_factor.fill_(scale_factor)
        if firing_ratio_record:
            firing_ratios.append(self.firing_ratio())

    def firing_ratio(self):
        firing_ratio = torch.mean(self.data.view(self.timesteps, -1, *self.chw), 0)
        return firing_ratio

    def timestep_dim_tensor(self):
        return self.data.view(self.timesteps, -1, *self.chw)

    def size(self, *args):
        return self.data.size(*args)

    def view(self, *args):
        return SpikeTensor(self.data.view(*args), self.timesteps, self.scale_factor.view(*args[1:]))

    def to_float(self):
        assert self.scale_factor is not None
        firing_ratio = self.firing_ratio()
        scaled_float_tensor = firing_ratio * self.scale_factor.unsqueeze(0)
        return scaled_float_tensor

    def __str__(self):
        return f"Spiketensor T{self.timesteps} Shape({self.b} {self.chw}) ScaleFactor {self.scale_factor} \n{self.data.shape}"
