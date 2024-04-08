import os
import datetime
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from nets.UDTransNet import UDTransNet
from nets.TF_configs import get_model_config
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch

torch.manual_seed(3407)

if __name__ == "__main__":
    TrainName = "240402_UDTrans_JinS_256_"
    Cuda = True
    seed = 3407
    sync_bn = False
    fp16 = False
    num_classes = 1 + 1
    backbone = "UDTransNet"

    pretrained = False

    model_path = ""
    input_shape = [256, 256]

    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 2
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 8

    Freeze_Train = False

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    Init_lr = 1e-4  # adam Init_lr=1e-4 or sgd Init_lr=1e-2 模型的最大学习率
    Min_lr = Init_lr * 0.01  # 模型的最小学习率，默认为最大学习率的0.01

    optimizer_type = "adam"  # adam Init_lr=1e-4 or sgd Init_lr=1e-2
    momentum = 0.9  # 优化器内部使用到的momentum参数
    # -------------------------------------------------------------------------------------------------------------------------------------------------
    weight_decay = 0  # 权值衰减，可防止过拟合, adam会导致weight_decay错误，使用adam时建议设置为0。
    # -------------------------------------------------------------------------------------------------------------------------------------------------

    lr_decay_type = "cos"  # 使用到的学习率下降方式，可选的有'step'、'cos'

    save_period = 10  # 多少个epoch保存一次权值

    save_dir = "../_logs-landslides"  # 权值与日志文件保存的文件夹

    eval_flag = True  # 是否在训练时进行评估，评估对象为验证集
    eval_period = 5  # 代表多少个epoch评估一次，不建议频繁的评估

    VOCdevkit_path = "VOCdevkit"  # 数据集路径

    dice_loss = True  # 是否使用dice loss来防止正负样本不平衡
    focal_loss = False  # 是否使用focal loss来防止正负样本不平衡
    Lovasz_Loss = False  # kaggle常用loss
    # ------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)

    num_workers = 16  # 多线程读取数据所使用的线程数, 1代表关闭多线程

    seed_everything(seed)
    ngpus_per_node = torch.cuda.device_count()
    # 只有单张显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    rank = 0
    if backbone == "UDTransNet":
        config_vit = get_model_config()
        # logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        # logger.info('transformer head dim: {}'.format(config_vit.transformer.embedding_channels))
        # logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        # logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = UDTransNet(config_vit, n_channels=14, n_classes=2, img_size=256)
    else:
        model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    # ----------------------#
    #   记录Loss
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        log_dir = os.path.join(save_dir, "loss_" + TrainName + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes,
            backbone=backbone,
            model_path=model_path,
            input_shape=input_shape,
            Init_Epoch=Init_Epoch,
            UnFreeze_Epoch=UnFreeze_Epoch,
            Unfreeze_batch_size=Unfreeze_batch_size,
            Init_lr=Init_lr,
            Min_lr=Min_lr,
            optimizer_type=optimizer_type,
            momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period,
            save_dir=save_dir,
            num_workers=num_workers,
            num_train=num_train,
            num_val=num_val,
        )
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        # ----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
            print(
                "\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"
                % (num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step)
            )
            print(
                "\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"
                % (total_step, wanted_step, wanted_epoch)
            )
    if True:
        batch_size = Unfreeze_batch_size

        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == "adam" else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == "adam" else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        optimizer = {
            "adam": optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            "sgd": optim.SGD(
                model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay
            ),
        }[optimizer_type]

        #   获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        #   判断每一个世代的长度
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        train_sampler = None
        val_sampler = None
        shuffle = True

        gen = DataLoader(
            train_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=unet_dataset_collate,
            sampler=train_sampler,
            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
        )
        gen_val = DataLoader(
            val_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=unet_dataset_collate,
            sampler=val_sampler,
            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
        )

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(
                model,
                input_shape,
                num_classes,
                val_lines,
                VOCdevkit_path,
                log_dir,
                Cuda,
                eval_flag=eval_flag,
                period=eval_period,
            )
        else:
            eval_callback = None

        #   开始模型训练
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and Freeze_Train:
                #   判断当前batch_size，自适应调整学习率
                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type == "adam" else 1e-1
                lr_limit_min = 1e-4 if optimizer_type == "adam" else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.unfreeze_backbone()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
                gen = DataLoader(
                    train_dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=unet_dataset_collate,
                    sampler=train_sampler,
                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                )
                gen_val = DataLoader(
                    val_dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=unet_dataset_collate,
                    sampler=val_sampler,
                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                )

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(
                model_train,
                model,
                loss_history,
                eval_callback,
                optimizer,
                epoch,
                epoch_step,
                epoch_step_val,
                gen,
                gen_val,
                UnFreeze_Epoch,
                Cuda,
                dice_loss,
                focal_loss,
                Lovasz_Loss,
                cls_weights,
                num_classes,
                fp16,
                scaler,
                save_period,
                save_dir,
                local_rank,
                TrainName,
                log_dir,
            )

        if local_rank == 0:
            loss_history.writer.close()
