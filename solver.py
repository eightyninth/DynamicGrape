# coding:utf-8
"""
Time: 2022/4/28 下午4:48
Author: eightyninth
File: solver.py
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import torch_geometric
from torch.utils.tensorboard import SummaryWriter
import os
from models.model import *
from utils.util import *
from dataset.data_load_for_gnn import *


def train(dataset, dataset_loader):
    backbone.train()
    damagedOrNot_model.train()
    biTrans_model.train()
    gnn_model.train()
    impute_model.train()
    l1_pred_list, l1_recover_list = [], []
    mse_pred_list, mse_recover_list = [], []
    dama_acc_list = []

    # {word: hde_code}
    character_dict = dataset.character_dict
    # [component(word)]
    component_equal_word = dataset.component_equal_word
    # {dim: [words]}
    component_in_word = dataset.component_in_word
    # {word: [dims]}
    word_has_component = dataset.word_has_component

    t1 = time.time()
    # Data(x=nodes, edge_index=edge_indexs, edge_attr=edge_attrs,
    #      unknown_index=torch.stack([start, end], dim=1)), area_codes, \
    # area_dims, words, damas, imgs
    for i, (data, area_code, area_dim, words, damas, imgs) in enumerate(iter(dataset_loader)):

        optimizer.zero_grad()

        feat_map = backbone(imgs.to(device))

        isDamaged_pred, _ = damagedOrNot_model(feat_map)

        """
        0: 表示完整
        1: 表示残损
        """
        isDamaged = torch.argmax(isDamaged_pred, dim=1)

        x, edge_attr, edge_index, unknown_index = data.x, data.edge_attr, data.edge_index, data.unknown_index

        label_index, label_attr, _, _ = make_label(unknown_index, words, component_in_word,
                                                   word_has_component, character_dict, isDamaged)

        # hde_pred, area_recover = biTrans_model(area_code.to(device))
        hde_pred, feat_recover = biTrans_model(feat_map.to(device))

        attr_instead = hde_pred[area_dim[0, :], area_dim[1, :]]
        double_attr_instead = torch.cat([attr_instead, attr_instead], dim=0)
        edge_attr[-double_attr_instead.shape[0]:, 0] = double_attr_instead

        x_embed = gnn_model(x.to(device), edge_attr.to(device), edge_index.to(device))
        X = impute_model([x_embed[label_index[0]], x_embed[label_index[1]]])

        if config.hyperparams.loss_type == "ce":
            # X_train = X[:int(train_edge_attr.shade[0] / 2)]
            X_train = X[:]
        else:
            # X_train = X[:int(train_edge_attr.shade[0] / 2), 0]
            X_train = X[:, 0]

        # X_train[known_mask] = train_labels[known_mask]

        loss_damaged = F.cross_entropy(isDamaged_pred, damas.cuda())

        if config.hyperparams.loss_type == "ce":
            loss_pred = F.cross_entropy(X_train, label_attr.cuda())
        elif config.hyperparams.loss_type == "mse_mse":
            # pred
            # loss_pred = F.mse_loss(X_train, torch.squeeze(label_edge_attr.cuda()))
            loss_pred = F.mse_loss(X_train, label_attr.cuda())
            # loss_pred = F.mse_loss(X_train, label_edge_attr.cuda(device_ids[0]))
            # mse = torch.mean((X_train.T - train_label_attr.cuda()) ** 2)

            # recover
            # loss_recover = F.mse_loss(area_recover.cpu()[area_dim[0, :], area_dim[1, :]],
            #                           area_code[area_dim[0, :], area_dim[1, :]])
            loss_recover = F.mse_loss(feat_recover,  feat_map)
        elif config.hyperparams.loss_type == "mse":
            # pred
            loss_pred = F.mse_loss(X_train, label_attr.cuda())

        if config.hyperparams.loss_type == "ce":
            loss = loss_pred + loss_damaged
        elif config.hyperparams.loss_type == "mse_mse":
            loss = loss_pred + loss_recover + loss_damaged
        else:
            loss = loss_pred + loss_damaged

        loss.backward()
        optimizer.step()

        l1_pred_list += F.l1_loss(X_train.detach().cpu(), label_attr, reduction="none").tolist()
        mse_pred_list += ((X_train.T.detach().cpu() - label_attr) ** 2).tolist()
        l1_recover_list += torch.mean(F.l1_loss(feat_recover, feat_map, reduction="none"), dim=1).tolist()
        mse_recover_list += torch.mean(F.mse_loss(feat_recover, feat_map, reduction="none"), dim=1).tolist()
        # l1_recover_list += F.l1_loss(area_recover.cpu()[area_dim[0, :], area_dim[1, :]],
        #                              area_code[area_dim[0, :], area_dim[1, :]], reduction="none").tolist()
        # mse_recover_list += F.mse_loss(area_recover.cpu()[area_dim[0, :], area_dim[1, :]],
        #                                area_code[area_dim[0, :], area_dim[1, :]], reduction="none").tolist()
        dama_acc_list += (isDamaged.cpu() == damas).tolist()

    t2 = time.time()
    total_t = t2 - t1
    return total_t, sum(dama_acc_list) / len(dama_acc_list), sum(mse_pred_list) / len(mse_pred_list), \
           sum(l1_pred_list) / len(l1_pred_list), sum(mse_recover_list) / len(mse_recover_list), \
           sum(l1_recover_list) / len(l1_recover_list)


def test(dataset, dataset_loader, acc_fun=None):
    backbone.eval()
    damagedOrNot_model.eval()
    biTrans_model.eval()
    gnn_model.eval()
    impute_model.eval()
    l1_pred_list, l1_recover_list = [], []
    mse_pred_list, mse_recover_list = [], []
    dama_acc_list = []
    if acc_fun: acc_list, is_dama_label = [], []

    # {word: hde_code}
    character_dict = dataset.character_dict
    # [component(word)]
    component_equal_word = dataset.component_equal_word
    # {dim: [words]}
    component_in_word = dataset.component_in_word
    # {word: [dims]}
    word_has_component = dataset.word_has_component

    t1 = time.time()
    with torch.no_grad():
        # Data(x=nodes, edge_index=edge_indexs, edge_attr=edge_attrs,
        #      unknown_index=torch.stack([start, end], dim=1)), area_codes, \
        # area_dims, words, damas, imgs
        for i, (data, area_code, area_dim, words, damas, imgs) in enumerate(iter(dataset_loader)):

            feat_map = backbone(imgs.to(device))

            isDamaged_pred, _ = damagedOrNot_model(feat_map)

            """
            0: 表示完整
            1: 表示残损
            """
            isDamaged = torch.argmax(isDamaged_pred, dim=1)

            x, edge_attr, edge_index, unknown_index = data.x, data.edge_attr, data.edge_index, data.unknown_index

            if acc_fun:
                label_index, label_attr, label, label_candidate = make_label(unknown_index, words,
                                                                             component_in_word,
                                                                             word_has_component, character_dict,
                                                                             isDamaged, has_acc=acc_fun is not None)
            else:
                label_index, label_attr, _, _ = make_label(unknown_index, words,
                                                           component_in_word,
                                                           word_has_component, character_dict,
                                                           isDamaged, has_acc=acc_fun is not None)

            # hde_pred, area_recover = biTrans_model(area_code.to(device))
            hde_pred, feat_recover = biTrans_model(feat_map.to(device))
            attr_instead = hde_pred[area_dim[0, :], area_dim[1, :]]
            double_attr_instead = torch.cat([attr_instead, attr_instead], dim=0)
            edge_attr[-double_attr_instead.shape[0]:, 0] = double_attr_instead

            x_embed = gnn_model(x.to(device), edge_attr.to(device), edge_index.to(device))
            X = impute_model([x_embed[label_index[0]], x_embed[label_index[1]]])

            X_output = X[:, 0]

            l1_pred_list += F.l1_loss(X_output.detach().cpu(), label_attr, reduction="none").tolist()
            mse_pred_list += ((X_output.T.detach().cpu() - label_attr) ** 2).tolist()

            l1_recover_list += torch.mean(F.l1_loss(feat_recover, feat_map, reduction="none"), dim=1).tolist()
            mse_recover_list += torch.mean(F.mse_loss(feat_recover, feat_map, reduction="none"), dim=1).tolist()
            # l1_recover_list += F.l1_loss(area_recover.cpu()[area_dim[0, :], area_dim[1, :]],
            #                              area_code[area_dim[0, :], area_dim[1, :]], reduction="none").tolist()
            # mse_recover_list += F.mse_loss(area_recover.cpu()[area_dim[0, :], area_dim[1, :]],
            #                                area_code[area_dim[0, :], area_dim[1, :]], reduction="none").tolist()

            dama_acc_list += (isDamaged.cpu() == damas).tolist()

            if acc_fun:
                is_dama_label += damas.tolist()

                acc = acc_fun(X_output.detach().cpu(), label_index, label, label_candidate)
                acc_list += acc

    t2 = time.time()
    total_t = t2 - t1
    if not acc_fun:
        acc_all = 0.
        acc_dama = 0.
        acc = 0.
    else:
        acc_numpy = np.array(acc_list)
        is_dama_label = np.array(is_dama_label)

        acc_all_numpy = acc_numpy[is_dama_label == 0]
        acc_dama_numpy = acc_numpy[is_dama_label == 1]

        acc_all = np.mean(acc_all_numpy)
        acc_dama = np.mean(acc_dama_numpy)
        acc = np.mean(acc_numpy)

    return total_t, sum(dama_acc_list) / len(dama_acc_list), sum(mse_pred_list) / len(mse_pred_list), \
           sum(l1_pred_list) / len(l1_pred_list), sum(mse_recover_list) / len(mse_recover_list), \
           sum(l1_recover_list) / len(l1_recover_list), acc_all, acc_dama, acc  # l1.detach().cpu()


if __name__ == "__main__":
    # 加载参数
    args = get_args("config/easage.json")
    config = process_config(args)
    print(config)

    # 设置训练device: cpu or gpu, 固定种子
    device = "cpu"
    if config.get("seed") is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available() and config.device == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.benchmark = True
            if config.device:
                device = config.device

    # 加载dataset
    dataset_train = character2radical(set_root=config.dataset.path, is_train="train")
    dataset_val = character2radical(set_root=config.dataset.path, is_train="val")
    dataset_test = character2radical(set_root=config.dataset.path, is_train="test")

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=config.dataset.batch_size, shuffle=True,
                                  collate_fn=dataset_train.collate_fn, drop_last=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=config.dataset.batch_size, shuffle=False,
                                collate_fn=dataset_train.collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=config.dataset.batch_size, shuffle=False,
                                 collate_fn=dataset_train.collate_fn)

    # 构建model
    backbone = Backbone(config.backbone)

    backbone = backbone.to(device)

    if config.damagedOrNot.type == "Linear":
        damagedOrNot_model = Linear(config.damagedOrNot)

    damagedOrNot_model = damagedOrNot_model.to(device)

    if config.biTransfer.type == "Linear":
        biTrans_model = Linear(config.biTransfer)

    biTrans_model = biTrans_model.to(device)

    if config.gnn.type == "EGSAGE_EGSAGE_EGSAGE":
        gnn_model = EGSAGEStack(config.gnn)

    gnn_model = gnn_model.to(device)

    impute_model = MLPNet(config)

    impute_model = impute_model.to(device)

    trainable_parameters = list(backbone.parameters()) + list(damagedOrNot_model.parameters()) + \
                           list(biTrans_model.parameters()) + list(gnn_model.parameters()) + list(
                           impute_model.parameters())

    scheduler, optimizer, lr = build_optimizer(config, trainable_parameters)

    print("train nums: {:d}, val nums: {:d}, test nums: {:d}.".format(len(dataset_train.mask_list),
                                                                      len(dataset_val.mask_list),
                                                                      len(dataset_test.mask_list)))

    if config.training:
        start_epoch = 0
        if config.hyperparams.resume:
            if config.hyperparams.resume.get("backbone"):
                start_epoch = load_model(backbone, config.hyperparams.resume["backbone"])
            if config.hyperparams.resume.get("damaged"):
                start_epoch = load_model(damagedOrNot_model, config.hyperparams.resume["damaged"])
            if config.hyperparams.resume.get("biTransfer"):
                start_epoch = load_model(biTrans_model, config.hyperparams.resume["biTransfer"])
            if config.hyperparams.resume.get("gnn"):
                start_epoch = load_model(gnn_model, config.hyperparams.resume["gnn"])
            if config.hyperparams.resume.get("imputation"):
                start_epoch = load_model(impute_model, config.hyperparams.resume["imputation"])

        valid_rmse, valid_l1, best_valid_rmse, best_valid_rmse_epoch, best_valid_l1, best_valid_l1_epoch, \
            = [], [], np.inf, 0, np.inf, 0

        if config.hyperparams.visualization:  # 进行可视化
            writer = SummaryWriter(log_dir="log_{}".format(config.gnn.name))
            tags = ["lr", "train_dama_acc", "train_mse_pred", "train_l1_pred", "train_mse_recover", "train_l1_recover",
                    "test_dama_acc", "test_mse_pred", "test_l1_pred", "test_mse_recover", "test_l1_recover",
                    "valid_dama_acc", "valid_mse_pred", "valid_l1_pred", "valid_mse_recover", "valid_l1_recover"]

        print("Training starting at epoch {:d}".format(start_epoch + 1))
        file = open(
            "{}_{}_result.txt".format(config.dataset.name, config.gnn.name), "a+")
        t_start = time.time()

        for epoch in range(start_epoch + 1, config.hyperparams.epochs + 1):
            t1 = time.time()

            tr_time, tr_dama_acc, tr_mse_pred, tr_l1_pred, tr_mse_recover, tr_l1_recover = train(dataset_train, dataloader_train)

            if scheduler:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]

            va_time, va_dama_acc, va_mse_pred, va_l1_pred, va_mse_recover, va_l1_recover, _, _, _ = test(dataset_val, dataloader_val)
            if va_l1_pred < best_valid_l1:
                best_valid_l1 = va_l1_pred
                best_valid_l1_epoch = epoch
                save_model(backbone, epoch, lr=lr, mode="l1", optimizer=optimizer,
                           model_name=config.backbone.name)
                save_model(damagedOrNot_model, epoch, lr=lr, mode="l1", optimizer=optimizer,
                           model_name=config.damagedOrNot.name)
                save_model(biTrans_model, epoch, lr=lr, mode="l1", optimizer=optimizer,
                           model_name=config.biTransfer.name)
                save_model(gnn_model, epoch, lr=lr, mode="l1", optimizer=optimizer, model_name=config.gnn.name)
                save_model(impute_model, epoch, lr=lr, mode="l1", optimizer=optimizer, model_name="imputation")
            if va_mse_pred < best_valid_rmse:
                best_valid_rmse = va_mse_pred
                best_valid_rmse_epoch = epoch
                save_model(backbone, epoch, lr=lr, mode="mse", optimizer=optimizer,
                           model_name=config.backbone.name)
                save_model(damagedOrNot_model, epoch, lr=lr, mode="mse", optimizer=optimizer,
                           model_name=config.damagedOrNot.name)
                save_model(biTrans_model, epoch, lr=lr, mode="mse", optimizer=optimizer,
                           model_name=config.biTransfer.name)
                save_model(gnn_model, epoch, lr=lr, mode="mse", optimizer=optimizer, model_name=config.gnn.name)
                save_model(impute_model, epoch, lr=lr, mode="mse", optimizer=optimizer, model_name="imputation")

            te_time, te_dama_acc, te_mse_pred, te_l1_pred, te_mse_recover, te_l1_recover, _, _, _ = test(dataset_test, dataloader_test)

            if config.hyperparams.visualization:
                writer.add_scalar(tags[0], lr, epoch)
                writer.add_scalars("dama_acc", {tags[1]: tr_mse_pred, tags[6]: te_mse_pred,
                                                tags[11]: va_mse_pred}, epoch)
                writer.add_scalars("mse_pred", {tags[2]: tr_mse_pred, tags[7]: te_mse_pred,
                                                tags[12]: va_mse_pred}, epoch)
                writer.add_scalars("l1_pred", {tags[3]: tr_l1_pred, tags[8]: te_l1_pred,
                                               tags[13]: va_l1_pred}, epoch)
                writer.add_scalars("mse_recover", {tags[4]: tr_mse_recover, tags[9]: \
                    te_mse_recover, tags[14]: va_mse_recover}, epoch)
                writer.add_scalars("l1_recover", {tags[5]: tr_l1_recover, tags[10]: \
                    te_l1_recover, tags[15]: va_l1_recover}, epoch)

            t2 = time.time()
            total_time = t2 - t1

            print("{:d}:{:d} process total_time:{:.6f} train total_time:{:.6f} val total_time: {:.6f} "
                  "test total_time:{:.6f} lr:{}\nTrain dama accuracy:{:.6f} Train mse_pred:{:.6f} Train l1_pred:{:.6f} Train "
                  "mse_recover:{:.6f} Train l1_recover:{:.6f}\nVal dama accuracy:{:.6f} Val mse_pred:{:.6f} Val l1_pred:{:.6f} "
                  "Val mse_recover:{:.6f} Val l1_recover:{:.6f}\nTest dama accuracy:{:.6f} Test mse_pred:{:.6f} Test l1_pred:{:.6f} "
                  "Test mse_recover:{:.6f} Test l1_recover:{:.6f}.".format(
                epoch, config.hyperparams.epochs, total_time, tr_time, va_time, te_time, lr,
                tr_dama_acc, tr_mse_pred, tr_l1_pred, tr_mse_recover, tr_l1_recover,
                va_dama_acc, va_mse_pred, va_l1_pred, va_mse_recover, va_l1_recover,
                te_dama_acc, te_mse_pred, te_l1_pred, te_mse_recover, te_l1_recover))
            print("{:d}:{:d} process total_time:{:.6f} train total_time:{:.6f} val total_time: {:.6f} "
                  "test total_time:{:.6f} lr:{} Train dama accuracy:{:.6f} Train mse_pred:{:.6f} Train l1_pred:{:.6f} Train "
                  "mse_recover:{:.6f} Train l1_recover:{:.6f} Val dama accuracy:{:.6f} Val mse_pred:{:.6f} Val l1_pred:{:.6f} "
                  "Val mse_recover:{:.6f} Val l1_recover:{:.6f} Test dama accuracy:{:.6f} Test mse_pred:{:.6f} Test l1_pred:{:.6f} "
                  "Test mse_recover:{:.6f} Test l1_recover:{:.6f}.".format(
                epoch, config.hyperparams.epochs, total_time, tr_time, va_time, te_time, lr,
                tr_dama_acc, tr_mse_pred, tr_l1_pred, tr_mse_recover, tr_l1_recover,
                va_dama_acc, va_mse_pred, va_l1_pred, va_mse_recover, va_l1_recover,
                te_dama_acc, te_mse_pred, te_l1_pred, te_mse_recover, te_l1_recover), file=file)

        file.close()
        t_end = time.time()
        print("Training End and use time {:.6f}.".format(t_end - t_start))

    else:
        load_epoch = 0
        if config.hyperparams.resume:
            if config.hyperparams.resume.get("backbone"):
                load_epoch = load_model(backbone, config.hyperparams.resume["backbone"])
            if config.hyperparams.resume.get("damaged"):
                load_epoch = load_model(damagedOrNot_model, config.hyperparams.resume["damaged"])
            if config.hyperparams.resume.get("biTransfer"):
                load_epoch = load_model(biTrans_model, config.hyperparams.resume["biTransfer"])
            if config.hyperparams.resume.get("gnn"):
                load_epoch = load_model(gnn_model, config.hyperparams.resume["gnn"])
            if config.hyperparams.resume.get("imputation"):
                load_epoch = load_model(impute_model, config.hyperparams.resume["imputation"])
        else:
            load_best_mse_paths = ["save_model_mse_No_acc/{}_{}.pth".format(config.backbone.name, load_epoch),
                                   "save_model_mse_No_acc/{}_{}.pth".format(config.damagedOrNot.name, load_epoch),
                                   "save_model_mse_No_acc/{}_{}.pth".format(config.biTransfer.name, load_epoch),
                                   "save_model_mse_No_acc/{}_{}.pth".format(config.gnn.name, load_epoch),
                                   "save_model_mse_No_acc/imputation_{}.pth".format(config.gnn.name, load_epoch)
                                   ]

            for _, (model, param_path) in enumerate(zip([biTrans_model, gnn_model, impute_model], load_best_mse_paths)):
                load_model(model, param_path)

        tr_time, tr_dama_acc, tr_mse_pred, tr_l1_pred, tr_mse_recover, tr_l1_recover, tr_acc_all,\
        tr_acc_dama, tr_acc = test(dataset_train, dataloader_train, acc_addgt)
        va_time, va_dama_acc, va_mse_pred, va_l1_pred, va_mse_recover, va_l1_recover, va_acc_all,\
        va_acc_dama, va_acc = test(dataset_val, dataloader_val, acc_addgt)
        te_time, te_dama_acc, te_mse_pred, te_l1_pred, te_mse_recover, te_l1_recover, te_acc_all,\
        te_acc_dama, te_acc = test(dataset_test, dataloader_test, acc_addgt)
        print("Load model at epoch {:d}: train total_time:{:.6f} val total_time: {:.6f} test total_time:{:.6f} lr:{} "
              "Train dama accuracy：{:.6f} Train mse_pred:{:.6f} Train l1_pred:{:.6f} Train mse_recover:{:.6f} Train "
              "l1_recover:{:.6f} Train all hde accuracy:{:.6f} Train dama hde accuracy:{:.6f} Train hde accuracy:{:.6f}"
              " Val dama accuracy：{:.6f} Val mse_pred:{:.6f} Val l1_pred:{:.6f} Val mse_recover:{:.6f} Val l1_recover:"
              "{:.6f} Val all hde accuracy:{:.6f} Val dama hde accuracy:{:.6f} Val hde accuracy:{:.6f} Test "
              "dama accuracy：{:.6f} Test mse_pred:{:.6f} Test l1_pred:{:.6f} Test mse_recover:{:.6f} Test l1_recover"
              ":{:.6f} Test all hde accuracy:{:.6f} Test dama hde accuracy:{:.6f} Test hde_acc:{:.6f}. ".format(
            load_epoch, tr_time, va_time, te_time, lr,
            tr_dama_acc, tr_mse_pred, tr_l1_pred, tr_mse_recover, tr_l1_recover, tr_acc_all, tr_acc_dama, tr_acc,
            va_dama_acc, va_mse_pred, va_l1_pred, va_mse_recover, va_l1_recover, va_acc_all, va_acc_dama, va_acc,
            te_dama_acc, te_mse_pred, te_l1_pred, te_mse_recover, te_l1_recover, te_acc_all, te_acc_dama, te_acc))
