# coding:utf-8
"""
Time: 2022/4/28 下午4:41
Author: eightyninth
File: data_load_for_gnn.py
加载部件掩码
"""
import sys

sys.path.append("..")

import string
import csv
import os
import imghdr
import cv2
import json
import torch
import networkx as nx
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# 设置选项允许重复加载动态链接库
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def distance_12(p1, p2):
    return np.sqrt(np.power((p2[0] - p1[0]), 2) + np.power((p2[1] - p1[1]), 2))


class character2radical(Dataset):
    def __init__(self, set_root="./hde-405", is_train="train", transform=None):
        super(character2radical, self).__init__()

        self.set_root = set_root
        self.is_train = is_train

        # 数据增强
        self.transform = transform

        # 读取字符编码与部首维度
        csv_path = os.path.join(self.set_root, "hde.csv")
        json_path = os.path.join(self.set_root, "radical_classes.json")

        self.component_dict = {}
        character_list = []
        with open(json_path) as f:
            component_dict = json.load(f)
        for k, v in component_dict.items():
            k = k.rstrip(string.digits)
            self.component_dict[k] = v

        self.character_dict = {}
        self.img_path = os.path.join(self.set_root, "photo")
        self.component_equal_word = []
        self.component_in_word = [list() for _ in range(len(self.component_dict))]
        self.word_has_component = {}

        with open(csv_path, encoding="gbk") as csv_f:
            csv_c = csv.reader(csv_f)
            for row in csv_c:
                code = [float(c) for c in row[1:]]
                code_index = np.nonzero(code)[0]
                if code_index.size <= 1:
                    self.component_equal_word.append(row[0])
                for code_idx in code_index:
                    self.component_in_word[code_idx].append(row[0])
                self.word_has_component.update({row[0]: code_index})
                self.character_dict.update({row[0]: code})
                character_list.append(code)

        # 读取图片
        with open(os.path.join(self.set_root, "all_mask_json/{}.json".format(self.is_train)), "r") as file:
            self.mask_list = json.load(file)

        # 构建基础图节点与边
        character_array = np.array(character_list, dtype=np.float32)
        character_num, component_num = character_array.shape

        # 节点
        x_component = torch.eye(component_num)
        x_character = torch.zeros(character_num, component_num)
        x_non_index = np.nonzero(character_array)
        x_character[x_non_index[0], x_non_index[1]] = 1.
        x = torch.cat((x_component, x_character), dim=0)

        # 边
        edge_nonzero_start, edge_nonzero_end = np.nonzero(character_array)
        edge_nonzero_single_attr = character_array[edge_nonzero_start, edge_nonzero_end]
        edge_nonzero_start += component_num
        edge_nonzero_attr = torch.from_numpy(
            np.expand_dims(np.hstack((edge_nonzero_single_attr, edge_nonzero_single_attr)), axis=1))
        edge_nonzero_single_idx = np.vstack((edge_nonzero_start, edge_nonzero_end))
        edge_nonzero_idx = torch.from_numpy(
            np.concatenate((edge_nonzero_start, edge_nonzero_end, edge_nonzero_end, edge_nonzero_start),
                           axis=0).reshape(2,
                                           -1)).long()

        self.x = x
        self.edge_index = edge_nonzero_idx
        self.edge_attr = edge_nonzero_attr

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_dict = self.mask_list[index]

        word = list(mask_dict.keys())[0]
        dama, masks = mask_dict[word]["dama"], mask_dict[word]["component"]
        components = []
        rect_list = []
        dims, areas = [], []
        imgs = None

        # 为了避免一个部件存在多个实例，转换字典为{mask: component}
        component_mask = {}
        for k, v in masks.items():
            for new_k in v:
                component_mask[new_k] = k

        for k, v in component_mask.items():
            # 读取图片 [路径, 矩形四角坐标, 矩形区域, 矩形中心]
            img = Image.open(os.path.join(self.img_path, k.replace("\\", "/")))

            if imgs is None:
                imgs = np.array(img.convert("L"))
            else:
                imgs += np.array(img.convert("L"))

            # img = Image.open(v).convert("1")
            # img.show()
            img_true = np.argwhere(np.array(img.convert("1")) == True)
            y_min, y_max, x_max, x_min = min(img_true[:, 0]), max(img_true[:, 0]), max(img_true[:, 1]), min(
                img_true[:, 1])
            lt, rt, rb, lb = [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
            area = (x_max - x_min) * (y_max - y_min)
            components.append([v, k, [lt, rt, rb, lb], area])
            rect_list.append([x_min, x_max, y_min, y_max])
            # 因为有重复出现的部件, area需要累加
            if self.component_dict[v] in dims:
                add_area_idx = dims.index(self.component_dict[v])
                areas[add_area_idx][0] += area
            else:
                dims.append(self.component_dict[v])
                areas.append([area])

        v_array = np.array(rect_list)
        x_all_min, x_all_max, y_all_min, y_all_max = min(v_array[:, 0]), \
                                                     max(v_array[:, 1]), min(v_array[:, 2]), max(v_array[:, 3])
        # 根据1: 1求矩形框面积
        all_max = max(x_all_max - x_all_min, y_all_max - y_all_min)
        area_sum = all_max ** 2

        # areas = np.squeeze(np.array(areas, dtype=np.float32))

        # 可视节点对应构建
        node_dim = len(self.component_dict)
        node = np.zeros((1, node_dim), dtype=np.float32)
        dims, areas = np.array(dims), np.array(areas, dtype=np.float32)
        areas /= area_sum
        node[0, dims] = 1.

        # area_code构建
        area_code = np.zeros(len(self.component_dict), dtype=np.float32)
        area_code[dims] = np.squeeze(areas)

        imgs = np.stack([imgs, imgs, imgs])
        # imgs = np.transpose(imgs, (1, 2, 0))
        # plt.imshow(imgs)
        # plt.show()
        # 字符, 残损标志, 新增节点, 拥有部件对应的维度(area_code非零维度), 边权值(area_code非零值), area_code, imgs
        return word, dama, node, dims, areas, area_code, imgs

    def collate_fn(self, batch):
        # 字符, 残损标志, 新增节点, 拥有部件对应的维度(area_code非零维度), 边权值(area_code非零值), area_code
        # word, dama, node, dims, areas
        words, damas = [], []
        x_add, indexs, attr = [], [], []
        area_codes, imgs = [], []
        for word, dama, node, dims, areas, area_code, img in batch:
            words.append(word)
            damas.append(dama)
            x_add.append(node)
            indexs.append(dims)
            attr.append(areas)
            area_codes.append(area_code)
            imgs.append(img)

        # 节点处理
        x_add = np.vstack(x_add)
        x_add = torch.from_numpy(x_add)
        nodes = torch.cat([self.x, x_add], dim=0)

        # 边处理
        character_dim, _ = self.x.shape
        start, end = [], []
        for index in indexs:
            for idx in index:
                start.append(character_dim)
                end.append(idx)
            character_dim += 1
        start, end = np.array(start), np.array(end)
        start, end = torch.from_numpy(start), torch.from_numpy(end)
        index_add = torch.cat([start, end, end, start], dim=0).reshape(2, -1).long()
        edge_indexs = torch.cat([self.edge_index, index_add], dim=1)
        attr_add = np.vstack(attr)
        attr_add = torch.from_numpy(attr_add)
        attr_add = torch.cat([attr_add, attr_add], dim=0)
        edge_attrs = torch.cat([self.edge_attr, attr_add], dim=0)

        area_codes = torch.FloatTensor(area_codes)
        area_dims = torch.cat([start, end], dim=0).reshape(2, -1)
        area_dims[0, :] -= self.x.shape[0]

        damas = torch.LongTensor(damas)

        # test = np.stack(imgs).astype(np.float32) / 255
        # img = test[0]
        # img = np.transpose(img, (1, 2, 0))
        # plt.imshow(img)
        # plt.show()

        imgs = torch.from_numpy(np.stack(imgs)).float() / 255
        # 构建的图, 待识别字符area_code, area_code中非零维度, 字符list, 残损标志, 图像
        return Data(x=nodes, edge_index=edge_indexs, edge_attr=edge_attrs,
                    unknown_index=torch.stack([start, end], dim=1)), area_codes, \
               area_dims, words, damas, imgs


if __name__ == "__main__":
    # pytorch 预定义均值与方差
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]

    dataset_train = character2radical(is_train="train", transform=None)
    dataset_val = character2radical(is_train="val", transform=None)
    dataset_test = character2radical(is_train="test", transform=None)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True,
                                  collate_fn=dataset_train.collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=4, shuffle=False, collate_fn=dataset_val.collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=4, shuffle=False, collate_fn=dataset_test.collate_fn)

    for i, (data, area_code, area_dim, word, dama, imgs) in enumerate(
            iter(dataloader_val)):
        # print(len(dataloader_train))
        img = imgs[0].numpy()
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.show()
        print(i)
        pass
    pass
