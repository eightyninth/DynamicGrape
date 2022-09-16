# coding:utf-8
"""
Time: 2022/4/28 下午4:48
Author: eightyninth
File: models.py
"""
import math

import torch
from utils.util import get_activation
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, zeros_
from torch_scatter import scatter_add
import torch.nn.functional as F
import utils.util as util
import numpy as np
from torchvision.models import resnet18

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class MLPNet(nn.Module):
    def __init__(self, config):
        super(MLPNet, self).__init__()
        if config.imputation.hidden == "":
            hiddens = []
        else:
            hiddens = list(map(int, config.imputation.hidden.split("_")))

        if config.gnn.concat_states:
            in_dim = config.gnn.hidden_node * len(list(config.gnn.type.split("_"))) * 2
        else:
            in_dim = config.gnn.hidden_node * 2

        layers = nn.ModuleList()

        for hidden in hiddens:
            layer = nn.Sequential(
                nn.Linear(in_dim, hidden),
                get_activation(config.imputation.activation),
                nn.Dropout(config.imputation.dropout)
            )
            layers.append(layer)
            in_dim = hidden
        layer = nn.Sequential(
            nn.Linear(in_dim, config.imputation.output),
            get_activation(config.imputation.output_activation)
        )
        layers.append(layer)
        self.layers = layers

    def forward(self, inputs):
        if torch.is_tensor(inputs):
            inputs = [inputs]
        input_var = torch.cat(inputs, -1)
        for layer in self.layers:
            input_var = layer(input_var)
        return input_var


class EGSAGEStack(nn.Module):
    def __init__(self, config):
        super(EGSAGEStack, self).__init__()
        self.dropout = config.dropout
        self.concat_states = config.concat_states
        self.activation = config.activation
        self.model_type = config.type.split("_")
        self.layers = len(self.model_type)
        if not config.norm_embs:
            self.norm_embs = [True] * self.layers
        else:
            self.norm_embs = list(map(bool, config.norm_embs.split("_")))
        if not config.node_post_mlp:
            self.node_post = [config.hidden_node]
        else:
            self.node_post = list(map(int, config.node_post_mlp.split("_")))

        self.convs = self.build_convs(config.init_node, config.init_edge, config.hidden_node,
                                      config.hidden_edge, config.edge_mode, config.type, config.aggregation)
        if config.concat_states:
            self.node_post_mlp = self.build_node_post_mlp(int(config.hidden_node * self.layers),
                                                          int(config.hidden_node * self.layers))
        else:
            self.node_post_mlp = self.build_node_post_mlp(config.hidden_node, config.hidden_node)

        self.edge_update_mlps = self.build_edge_update_mlps(config.hidden_node, config.init_edge, config.hidden_edge)

    def forward(self, x, edge_attr, edge_index):
        if self.concat_states:
            concat_x = []
        for l, (conv_name, conv) in enumerate(zip(self.model_type, self.convs)):
            if conv_name == "EGSAGE":
                x = conv(x, edge_attr, edge_index)
            else:
                x = conv(x, edge_index)
            if self.concat_states:
                concat_x.append(x)
            edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
        if self.concat_states:
            x = torch.cat(concat_x, 1)
        x = self.node_post_mlp(x)
        return x

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0], :]
        x_j = x[edge_index[1], :]
        edge_attr = mlp(torch.cat((x_i, x_j, edge_attr), dim=-1))
        return edge_attr

    def build_edge_update_mlps(self, node_dim, edge_in, edge_hid):
        edge_update_mlps = nn.ModuleList()
        edge_update_mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim + edge_in, edge_hid),
            get_activation(self.activation)
        )
        edge_update_mlps.append(edge_update_mlp)
        for l in range(1, self.layers):
            edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim + node_dim + edge_hid, edge_hid),
                get_activation(self.activation)
            )
            edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps

    def build_node_post_mlp(self, in_dim, out_dim, ):
        if 0 in self.node_post:
            return get_activation('none')
        else:
            layers = []
            for hidden_dim in self.node_post:
                layer = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    get_activation(self.activation),
                    nn.Dropout(self.dropout)
                )
                layers.append(layer)
                in_dim = hidden_dim
            layer = nn.Linear(in_dim, out_dim)
            layers.append(layer)
            return nn.Sequential(*layers)

    def build_convs(self, init_node, init_edge, hidden_node, hidden_edge, edge_mode, model_type, aggr):
        convs = nn.ModuleList()
        conv = self.build_conv_model(model_type[0], init_node, hidden_node, init_edge, edge_mode, self.norm_embs[0],
                                     aggr)
        convs.append(conv)
        for l in range(1, self.layers):
            conv = self.build_conv_model(model_type[l], hidden_node, hidden_node, hidden_edge, edge_mode,
                                         self.norm_embs[l], aggr)
            convs.append(conv)
        return convs

    def build_conv_model(self, model_type, in_node, out_node, in_edge, edge_mode, norm_emb, aggr):
        """
        目前只有添加边嵌入的GraphSAGE
        """
        # if model_type == "EGSAGE":
        return EGSAGE(in_node, out_node, in_edge, edge_mode, self.activation, norm_emb, aggr)


class EGSAGE(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, edge_mode, activation, norm_emb, aggr):
        super(EGSAGE, self).__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.edge_mode = edge_mode

        if edge_mode == 0:  # 做注意力处理
            self.message_lin = nn.Linear(in_channels, out_channels)
            self.attention_lin = nn.Linear(2 * in_channels + edge_channels, 1)
        elif edge_mode == 1:  # 直接拼接邻域节点和边
            self.message_lin = nn.Linear(in_channels + edge_channels, out_channels)
        elif edge_mode == 2:  # 直接拼接根节点，邻域节点和边
            self.message_lin = nn.Linear(2 * in_channels + edge_channels, out_channels)
        elif edge_mode == 3:  # 同样直接拼接根节点，邻域节点和边, 不同的是作2-MLP
            self.message_lin = nn.Sequential(
                nn.Linear(2 * in_channels + edge_channels, out_channels),
                get_activation(activation),
                nn.Linear(out_channels, out_channels)
            )

        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)

        self.message_activation = get_activation(activation)
        self.update_activation = get_activation(activation)
        self.norm_emb = norm_emb

    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))

    def message(self, x_i, x_j, edge_attr, edge_index, size):
        if self.edge_mode == 0:  # 做注意力处理
            attention = self.attention_lin(torch.cat((x_i, x_j, edge_attr), dim=-1))
            m_j = attention * self.message_activation(self.message_lin(x_j))
        elif self.edge_mode == 1:  # 直接拼接邻域节点和边
            m_j = torch.cat((x_j, edge_attr), dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 2 or self.edge_mode == 3:  # 直接拼接根节点，邻域节点和边  # 同样直接拼接根节点，邻域节点和边, 不同的是作2-MLP
            m_j = torch.cat((x_i, x_j, edge_attr), dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        return m_j

    def update(self, aggr_out, x):
        aggr_out = self.update_activation(self.agg_lin(torch.cat((aggr_out, x), dim=-1)))
        if self.norm_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out


class SMGStack(nn.Module):
    def __init__(self, config):
        super(SMGStack, self).__init__()

        # 源代码对节点初始特征维度进行处理
        # self.lin0 = nn.Linear(config.init_node, config.hidden_node)
        self.layers_num = len(list(config.type.split("_")))
        self.convs = nn.ModuleList()

        self.convs.append(SparseConv(config.init_node, config.hidden_node))
        for i in range(1, self.layers_num):
            self.convs.append(SparseConv(config.hidden_node, config.hidden_node))

        self.masks = nn.ModuleList()
        if config.multi_channels:
            out_channel = config.hidden_node
        else:
            out_channel = 1
        self.masks.append(WeightConv(config.init_node, config.hidden_node, out_channel))
        for i in range(1, self.layers_num):
            self.masks.append(WeightConv(config.hidden_node, config.hidden_node, out_channel))

        self.post_mlp1 = nn.Linear(config.hidden_node, config.hidden_node)
        self.post_mlp2 = nn.Linear(config.hidden_node, config.hidden_node)

    def forward(self, x, edge_attr, edge_index):
        # x = self.lin0(x)
        mask_val = None
        for i, conv in enumerate(self.convs):
            mask = self.masks[i]
            mask_val = mask(x, edge_index, mask_val, edge_attr)
            x = F.relu(conv(x, edge_index, mask_val, edge_attr))
        x = F.relu(self.post_mlp1(x))
        x = F.dropout(x, p=0.5)
        x = self.post_mlp2(x)
        return x


class SparseConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add", bias=False):
        super(SparseConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        pyg_nn.inits.uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()

    def forward(self, x, edge_index, mask, edge_weight=None, size=None):
        h = x * mask
        h = torch.matmul(h, self.weight)
        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight) * mask

    def message(self, h_j, edge_weight):
        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        return aggr_out + self.lin(x)


class WeightConv(MessagePassing):

    def __init__(self, in_channels, hid_channels, out_channels=1, aggr='add', bias=True, ):
        super(WeightConv, self).__init__(aggr=aggr)

        self.l1 = torch.nn.Linear(in_channels, hid_channels, bias=bias)
        self.l2 = torch.nn.Linear(in_channels, hid_channels, bias=bias)
        self.mlp1 = torch.nn.Linear(hid_channels * 2, hid_channels, bias=bias)
        self.mlp2 = torch.nn.Linear(hid_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()

    def forward(self, x, edge_index, mask=None, edge_weight=None, size=None):
        if mask is not None:
            x = x * mask
        h = self.l1(x)
        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)

    def message(self, h_j, edge_weight):
        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        weight = torch.cat([aggr_out, self.l2(x)], dim=-1)
        weight = F.relu(weight)
        weight = self.mlp1(weight)
        weight = F.relu(weight)
        weight = self.mlp2(weight)
        weight = torch.sigmoid(weight)
        return weight


class Linear(nn.Module):
    def __init__(self, config):
        super(Linear, self).__init__()

        self.W = nn.Parameter(torch.randn(config.in_channel, config.out_channel), requires_grad=True)

        self.drop = nn.Dropout(config.dropout)
        self.active = get_activation(config.activation)

        self.reset_params()

    def reset_params(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, code):
        pred = torch.matmul(code, self.W)
        recover = torch.matmul(pred, self.W.T)

        pred = self.active(self.drop(pred))
        recover = self.active(self.drop(recover))

        return pred, recover
        # return recover


class Backbone(nn.Module):
    def __init__(self, config):
        super(Backbone, self).__init__()

        if config.type == "resnet18":
            self.extractor = resnet18(pretrained=False)

        if config.pretrained:
            self.extractor.load_state_dict(torch.load(config.pretrained), strict=True)

    def forward(self, x):
        x = self.extractor.conv1(x)
        x = self.extractor.bn1(x)
        x = self.extractor.relu(x)
        x = self.extractor.maxpool(x)
        x = self.extractor.layer1(x)
        x = self.extractor.layer2(x)
        x = self.extractor.layer3(x)
        x = self.extractor.layer4(x)
        x = self.extractor.avgpool(x)
        return torch.squeeze(x)
