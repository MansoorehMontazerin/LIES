import torch
from preprocess import *
import math
from math import log, exp, sin
from torch import nn
import numpy as np
 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

threshold = -5
k_sin = 1

class LogActFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x0 = 5e-3
        ctx.save_for_backward(x)
        return torch.where(x > x0, torch.log(x), torch.log(torch.tensor(x0, device=x.device)))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x0 = 5e-3
        grad_input = torch.where(x > x0, 1/x, torch.zeros_like(x))
        return grad_output * grad_input

def log_act(x):
    return LogActFunction.apply(x)


class ExpActFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x0 = 4
        x1 = -20
        ctx.save_for_backward(x)
        return torch.where((x < x0), torch.exp(x), torch.exp(torch.tensor(x0, device=x.device)))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x0 = 4
        x1 = -20
        grad_input = torch.where(x < x0, torch.exp(x), torch.zeros_like(x))
        return grad_output * grad_input

def exp_act(x):
    return ExpActFunction.apply(x)

def exp_act_der(x):
    x0 = 500
    k = 10
    a = (-(k + 1) * x0 + math.log(k)) / k
    b = math.exp(x0) + math.exp(-k * (x0 + a))
    return torch.where(x < x0, torch.exp(x), k*torch.exp(-k * (x + a)))


class math_act(nn.Module):
    def __init__(self):
        super(math_act, self).__init__() 
        self.loss = 0

    def forward(self, x):
        assert (x.shape[-1] == 4)
        self.loss = 0
        x0, x1, x2, x3 = torch.chunk(x, 4, -1)
        mask = (x3 < 5e-3)
        if mask.any():
            self.loss += torch.abs(torch.tensor(5e-3) - x3[mask]).sum()
                                               
        x3 = log_act(x3)
        mask = x1 > 4
        if mask.any():
            self.loss += self.loss + torch.abs((torch.tensor(4)) - (x1[mask])).sum()
        x1 = exp_act(x1)
        x2 = k_sin*torch.sin(x2)
        return torch.cat((x0, x1, x2, x3), -1)


class math_actNoS(nn.Module):
    def __init__(self):
        super(math_actNoS, self).__init__()
        self.loss = 0
        
    def forward(self, x):
        assert (x.shape[-1] == 3)
        self.loss = 0
        x0, x1, x2 = torch.chunk(x, 3, -1)
        mask = (x2 < 5e-3)
        if mask.any():
            self.loss += self.loss + torch.abs(torch.tensor(5e-3) - x2[mask]).sum()
                                               
        x2 = log_act(x2)
        mask = x1 > 4
        if mask.any():
            self.loss += self.loss + torch.abs((torch.tensor(4)) - (x1[mask])).sum()
        x1 = exp_act(x1)
        return torch.cat((x0, x1, x2), -1)


class node_bridge(nn.Module):
    def __init__(self):
        super(node_bridge, self).__init__()
        self.node = nn.ModuleList()
        for i in range(4):
            self.node.append(nn.Linear(in_features=1, out_features=1, bias=False))

    def forward(self, x):
        assert (x.shape[-1] == 4)
        x0, x1, x2, x3 = torch.chunk(x, 4, -1)
        x0 = self.node[0](x0)
        x1 = self.node[1](x1)
        x2 = self.node[2](x2)
        x3 = self.node[3](x3)
        return torch.cat((x0, x1, x2, x3), -1)


class node_bridgeNoS(nn.Module):
    def __init__(self):
        super(node_bridgeNoS, self).__init__()
        self.node = nn.ModuleList()
        for i in range(3):
            self.node.append(nn.Linear(in_features=1, out_features=1, bias=False))

    def forward(self, x):
        assert (x.shape[-1] == 3)
        x0, x1, x2 = torch.chunk(x, 3, -1)
        x0 = self.node[0](x0)
        x1 = self.node[1](x1)
        x2 = self.node[2](x2)
        return torch.cat((x0, x1, x2), -1)


class math_layer(nn.Module):
    def __init__(self, input_dim=4):
        super(math_layer, self).__init__()
        self.fc = nn.Linear(in_features=input_dim, out_features=4, bias=False)
        self.act = math_act()
        self.activations = {}

    def forward(self, x):
        self.activations['input'] = x.detach()
        x = self.fc(x)
        self.activations['fc'] = x.detach()
        x = self.act(x)
        return x


class math_layerNoS(nn.Module):
    def __init__(self, input_dim=3):
        super(math_layerNoS, self).__init__()
        self.fc = nn.Linear(in_features=input_dim, out_features=3, bias=False)
        self.act = math_actNoS()
        self.activations = {}

    def forward(self, x):
        self.activations['input'] = x.detach()
        x = self.fc(x)
        self.activations['fc'] = x.detach()
        x = self.act(x)
        return x


class output_layer(nn.Module):
    def __init__(self):
        super(output_layer, self).__init__()
        self.fc = nn.Linear(4, 1, bias=True)

    def forward(self, x):
        return self.fc(x)

class output_layerNoS(nn.Module):
    def __init__(self):
        super(output_layerNoS, self).__init__()
        self.fc = nn.Linear(3, 1, bias=True)

    def forward(self, x):
        return self.fc(x)


class shallow_model(nn.Module):
    def __init__(self, inputs=3, hidden_layers=3):
        inputs += 1
        super(shallow_model, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(math_layer(inputs))
        for i in range(hidden_layers):
            self.model.append(math_layer())
        self.model.append(output_layer())

    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x


class jump_model(nn.Module):
    def __init__(self, inputs=3, hidden_layers=3):
        inputs += 1
        super(jump_model, self).__init__()
        self.model = nn.ModuleList()
        self.model.append(math_layer(inputs))
        for i in range(hidden_layers):
            self.model.append(math_layer())
        self.adjuster = nn.ModuleList()
        for i in range(hidden_layers + 2):
            for j in range(i):
                if j == 0:
                    self.adjuster.append(nn.Linear(inputs, 4, bias=False))
                else:
                    self.adjuster.append(nn.Linear(4, 4, bias=False))
        self.model.append(output_layer())


    def forward(self, x):
        inputs = [x]
        t = 0
        for i, m in enumerate(self.model):
            input_x = inputs[-1]
            for j in range(i):
                input_x += self.adjuster[t](inputs[j])
                t += 1
            inputs.append(m(input_x))
        return inputs[-1]


class jump_node_model(nn.Module):
    def __init__(self, inputs=3, hidden_layers=3):
        inputs += 1
        super(jump_node_model, self).__init__()
        self.model = nn.ModuleList()
        self.bridges = nn.ModuleList()
        self.model.append(math_layer(inputs))
        for i in range(hidden_layers):
            self.model.append(math_layer())
        for i in range(hidden_layers + 1):
            self.bridges.append(node_bridge())
        self.adjuster = nn.ModuleList()
        for i in range(hidden_layers + 2):
            for j in range(i):
                if j == 0:
                    self.adjuster.append(nn.Linear(inputs, 4, bias=False))
                else:
                    self.adjuster.append(nn.Linear(4, 4, bias=False))
        self.model.append(output_layer())

    def forward(self, x):
        inputs = [x]
        t = 0
        for i, m in enumerate(self.model):
            input_x = inputs[-1]
            for j in range(i):
                input_x += self.adjuster[t](inputs[j])
                t += 1
            if i == len(self.model) - 1:
                inputs.append(m(input_x))
            else:
                inputs.append(self.bridges[i](m(input_x)))
        return inputs[-1]


class jump_node_modelNoS(nn.Module):
    def __init__(self, inputs=3, hidden_layers=3):
        inputs += 1
        super(jump_node_modelNoS, self).__init__()
        self.model = nn.ModuleList()
        self.bridges = nn.ModuleList()
        self.model.append(math_layerNoS(inputs))
        for i in range(hidden_layers):
            self.model.append(math_layerNoS())
        for i in range(hidden_layers + 1):
            self.bridges.append(node_bridgeNoS())
        self.adjuster = nn.ModuleList()
        for i in range(hidden_layers + 2):
            for j in range(i):
                if j == 0:
                    self.adjuster.append(nn.Linear(inputs, 3, bias=False))
                else:
                    self.adjuster.append(nn.Linear(3, 3, bias=False))
        self.model.append(output_layerNoS())

    def forward(self, x):
        inputs = [x]
        t = 0
        for i, m in enumerate(self.model):
            input_x = inputs[-1]
            for j in range(i):
                input_x += self.adjuster[t](inputs[j])
                t += 1
            if i == len(self.model) - 1:
                inputs.append(m(input_x))
            else:
                inputs.append(self.bridges[i](m(input_x)))
        return inputs[-1]

