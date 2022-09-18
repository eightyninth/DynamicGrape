# coding:utf-8
"""
Time: 2022/4/28 下午4:48
Author: eightyninth
File: util.py
"""
import argparse
import os
import json
from easydict import EasyDict
import torch.optim as optim
from sklearn import preprocessing
import numpy as np
import torch
import csv
import torch.nn.functional as F
from torch_geometric.data import Data
from scipy.sparse import coo_matrix


def get_activation(activation):
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "prelu":
        return torch.nn.PReLU()
    elif activation == "softmax":
        return torch.nn.Softmax()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif not activation or activation == "None":
        return torch.nn.Identity()
    else:
        raise NotImplementedError


def get_args(config_path):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", default=config_path, help="The Configuration File Path")
    args = argparser.parse_args()
    return args


def process_config(arg):
    with open(arg.config, "r") as json_file:
        config_dict = json.load(json_file)
    config = EasyDict(config_dict)
    return config


def build_optimizer(config, parameters):
    weight_decay = config.optim.decay
    filter_fn = filter(lambda p: p.requires_grad, parameters)
    if config.optim.type == "adam":
        optimizer = optim.Adam(filter_fn, lr=config.optim.learning_rate, weight_decay=weight_decay)
    elif config.optim.type == "sgd":
        optimizer = optim.SGD(filter_fn, lr=config.optim.learning_rate, momentum=0.95, weight_decay=weight_decay)
    elif config.optim.type == "rmsprop":
        optimizer = optim.RMSprop(filter_fn, lr=config.optim.learning_rate, weight_decay=weight_decay)
    elif config.optim.type == "adagrad":
        optimizer = optim.Adagrad(filter_fn, lr=config.optim.learning_rate, weight_decay=weight_decay)

    if not config.optim.scheduler or config.optim.scheduler == 'none':
        return None, optimizer, config.optim.learning_rate
    elif config.optim.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.optim.decay_step,
                                              gamma=config.optim.decay_rate)
    elif config.optim.scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.hyperparams.epochs)
    return scheduler, optimizer, config.optim.learning_rate


def load_model(model, param_path):
    print("Loading from {}.".format(param_path))
    state_dict = torch.load(param_path)
    model.load_state_dict(state_dict["model"])
    epoch = state_dict["epoch"]
    print("Loading Finish.")
    return epoch


def save_model(model, epoch, lr, optimizer, model_name, mode="None", has_acc="No_acc"):
    save_path = os.path.join(".",
                             "save_model_{}_{}".format(mode, has_acc))
    if not os.path.exists(save_path): os.makedirs(save_path)
    if not lr: lr = 0.
    save_path = os.path.join(save_path, "{}_{}.pth".format(model_name, epoch))
    print("Saveing {} to {}".format(model_name, save_path))
    state_dict = {
        "lr": lr,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state_dict, save_path)


def make_label(unknown_index, words, component_in_word, word_has_component, character_dict, is_damaged, has_acc=False):
    known_index, refer_index, infer_index = unknown_index.numpy(), {}, {}
    for k_idx in known_index:
        if k_idx[0] not in refer_index.keys():
            refer_index[k_idx[0]] = [k_idx[1]]
        else:
            refer_index[k_idx[0]] += [k_idx[1]]

    # 按顺序排列字符
    refer_index_key = sorted(refer_index.keys())

    for _, (w, is_d, r_idx) in enumerate(zip(words, is_damaged, refer_index_key)):
        known = refer_index[r_idx]
        known2word, word2refer = None, set()
        for k in known:
            if known2word:
                known2word = known2word.intersection(set(component_in_word[k]))
            else:
                known2word = set(component_in_word[k])
        known2word = list(known2word)
        if is_d:
            for k2w in known2word:
                for w2c in word_has_component[k2w]:
                    word2refer.add(w2c)
        else:
            for w2c in word_has_component[w]:
                word2refer.add(w2c)
        infer_index[r_idx] = [w, known2word, list(word2refer)]

    start, end, attr = [], [], []
    if has_acc:
        label, label_candidate = [], []

    for r_idx in refer_index_key:
        word, word_candidate, end_index = infer_index[r_idx]
        start += [r_idx] * len(end_index)
        end += end_index
        code = np.array(character_dict[word])
        attr += code[end_index].tolist()
        if has_acc:
            label.append(word_candidate.index(word))
            candidate = []
            for w_c in word_candidate:
                candidate.append(character_dict[w_c])
            candidate = torch.FloatTensor(candidate)
            label_candidate.append(candidate)

    label_index = torch.FloatTensor([start, end]).long()
    label_attr = torch.FloatTensor(attr)

    if has_acc:
        return label_index, label_attr, label, label_candidate
    else:
        return label_index, label_attr, None, None


def acc_addgt(output, label_index, label, label_candidate):
    label_index_split = label_index[0].numpy()
    label_index_split_len = len(label_index_split)
    _, label_index_split = np.unique(label_index_split, return_index=True)
    label_index_split = np.append(label_index_split, label_index_split_len)
    acc = []
    for i, (l, l_c) in enumerate(zip(label, label_candidate)):
        component_dim = label_candidate[0].shape[-1]
        pred = torch.zeros(component_dim)
        pred[label_index[1, label_index_split[i]: label_index_split[i + 1]]] = \
            output[label_index_split[i]: label_index_split[i + 1]]
        pred = torch.unsqueeze(pred, dim=0)

        pred = pred.repeat(l_c.shape[0], 1)
        pred_dis = cos_distance(l_c, pred)
        pred_max = torch.argmin(pred_dis, dim=-1)
        acc.append((l == pred_max).int().item())
    return acc


def cos_distance(gt, pred):
    return 1 - torch.cosine_similarity(gt, pred, dim=-1)
