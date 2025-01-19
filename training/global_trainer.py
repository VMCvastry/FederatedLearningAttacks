import json
import os
from training.train_attack_utils import (
    compute_all_gradient_metrics,
    compute_losses,
    compute_losses_for_indexes,
)
from dataset.dataset_utils import *
import copy
from tqdm import tqdm
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
import time

import torch.nn as nn
import models as models

from .base import BaseTrainer
from .trainer_private import TrainerPrivate


class FederatedLearning(BaseTrainer):
    """
    Perform federated learning
    """

    def __init__(self, args):
        super().__init__(args)  # define many self attributes from args
        self.watch_train_client_id = 0
        self.watch_val_client_id = 1

        self.criterion = torch.nn.CrossEntropyLoss()
        self.in_channels = 3
        self.num_bit = args.num_bit
        self.num_trigger = args.num_trigger
        self.dp = args.dp
        self.defense = args.defense
        self.lr_up = args.lr_up
        self.sigma = args.sigma
        self.cosine_attack = args.cosine_attack
        self.sigma_sgd = args.sigma_sgd
        self.grad_norm = args.grad_norm
        self.save_dir = args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.data_root = args.data_root

        print("==> Preparing data...")
        (
            self.train_set,
            self.test_set,
            self.dict_users,
            self.train_idxs,
            self.val_idxs,
        ) = get_data(
            dataset=self.dataset,
            data_root=self.data_root,
            iid=self.iid,
            num_users=self.num_users,
            data_aug=args.data_augment,
            noniid_beta=args.beta,
        )

        if self.dataset == "cifar10":
            self.num_classes = 10
        elif self.dataset == "cifar100":
            self.num_classes = 100

        self.schedule_milestone = args.schedule_milestone

        self.MIA_trainset_dir = []
        self.MIA_valset_dir = []
        self.MIA_trainset_dir_cos = []
        self.MIA_valset_dir_cos = []
        self.MIA_mode = args.MIA_mode
        self.train_idxs_cos = []
        self.testset_idx = (50000 + np.arange(10000)).astype(
            int
        )  # The last 10,000 samples are used as the test set
        self.testset_idx_cos = (50000 + np.arange(1000)).astype(int)

        print("==> Preparing model...")

        self.logs = {
            "train_acc": [],
            "train_sign_acc": [],
            "train_loss": [],
            "val_acc": [],
            "val_loss": [],
            "test_acc": [],
            "test_loss": [],
            "keys": [],
            "best_test_acc": -np.inf,
            "best_model": [],
            "local_loss": [],
        }

        self.construct_model()

        self.w_t = copy.deepcopy(self.model.state_dict())

        self.trainer = TrainerPrivate(
            self.model,
            self.train_set,
            self.device,
            self.sigma,
            self.num_classes,
        )

        self.makedirs_or_load()

    def construct_model(self):

        model = models.__dict__[self.model_name](num_classes=self.num_classes)

        # model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

        torch.backends.cudnn.benchmark = True
        print("Total params: %.2f" % (sum(p.numel() for p in model.parameters())))

    def train(self):
        # these dataloader would only be used in calculating accuracy and loss
        train_ldr = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        val_ldr = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        test_ldr = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

        local_train_ldrs = []
        if self.iid:
            for i in range(self.num_users):
                local_train_ldr = DataLoader(
                    DatasetSplit(self.train_set, self.dict_users[i]),
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=2,
                )
                # print("len:",len(local_train_ldr)) 1
                local_train_ldrs.append(local_train_ldr)

        else:
            for i in range(self.num_users):
                local_train_ldr = DataLoader(
                    self.dict_users[i],
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=2,
                )
                local_train_ldrs.append(local_train_ldr)

        total_time = 0
        file_name = "_".join(
            [
                "a",
                self.model_name,
                self.dataset,
                str(self.num_users),
                "sgd",
                str(self.lr_up),
                str(self.batch_size),
                str(time.strftime("%Y_%m_%d_%H%M%S", time.localtime())),
            ]
        )

        b = os.path.join(os.getcwd(), self.save_dir)
        if not os.path.exists(b):
            os.makedirs(b)
        fn = b + "/" + file_name + ".log"
        fn = file_name + ".log"
        fn = os.path.join(b, fn)
        print("training log saved in:", fn)

        lr_0 = self.lr

        for epoch in range(self.epochs):

            global_state_dict = copy.deepcopy(self.model.state_dict())

            if self.sampling_type == "uniform":
                self.m = max(int(self.frac * self.num_users), 1)
                idxs_users = np.random.choice(
                    range(self.num_users), self.m, replace=False
                )

            (
                local_ws,
                local_losses,
            ) = (
                [],
                [],
            )

            start = time.time()
            for idx in tqdm(
                idxs_users, desc="Epoch:%d, lr:%f" % (self.epochs, self.lr)
            ):

                self.model.load_state_dict(global_state_dict)

                local_w, local_loss = self.trainer.local_update_noback(
                    local_train_ldrs[idx], self.local_ep, self.lr
                )
                test_loss, test_acc = self.trainer.test(val_ldr)

                local_ws.append(copy.deepcopy(local_w))
                local_losses.append(local_loss)

                if (
                    self.MIA_mode == 1
                    and (
                        (epoch + 1) % 10 == 0
                        or epoch == 0
                        or epoch in self.schedule_milestone
                        or epoch - 1 in self.schedule_milestone
                        or epoch - 2 in self.schedule_milestone
                    )
                    == 1
                ):
                    # Data that needs to be saved: the results of all clients for client0 and test set; client0 saves client0 data as train, and other clients as val
                    save_dict = {}
                    save_dict["test_acc"] = test_acc
                    save_dict["test_loss"] = test_loss
                    crossentropy_noreduce = nn.CrossEntropyLoss(reduction="none")

                    test_res = compute_losses(
                        test_ldr, self.model, crossentropy_noreduce, self.device
                    )
                    save_dict["test_index"] = self.testset_idx  # 10000
                    save_dict["test_res"] = test_res

                    # target -> self.watch_train_client_id=0
                    train_res = compute_losses_for_indexes(
                        self.train_set,
                        self.train_idxs[self.watch_train_client_id],
                        self.model,
                        crossentropy_noreduce,
                        self.device,
                    )
                    save_dict["train_index"] = self.train_idxs[
                        self.watch_train_client_id
                    ]
                    save_dict["train_res"] = train_res

                    # validation -> self.watch_val_client_id=1
                    val_res = compute_losses_for_indexes(
                        self.train_set,
                        self.train_idxs[self.watch_val_client_id],
                        self.model,
                        crossentropy_noreduce,
                        self.device,
                    )
                    save_dict["val_index"] = self.train_idxs[self.watch_val_client_id]
                    save_dict["val_res"] = val_res

                    if (
                        self.cosine_attack == True
                    ):  # and idx == self.watch_train_client_id:

                        ## compute model grads
                        model_grads = []
                        for name, local_param in self.model.named_parameters():
                            if local_param.requires_grad == True:
                                para_diff = local_w[name] - global_state_dict[name]
                                model_grads.append(para_diff.detach().cpu().flatten())
                        model_grads = torch.cat(model_grads, -1)

                        ## compute cosine score
                        cos_model = models.__dict__[self.model_name](
                            num_classes=self.num_classes
                        )
                        cos_model = cos_model.to(self.device)
                        cos_model.load_state_dict(
                            global_state_dict
                        )  # Load the basic global model
                        (
                            train_cos,
                            train_diffs,
                            train_norm,
                            val_cos,
                            val_diffs,
                            val_norm,
                            test_cos,
                            test_diffs,
                            test_norm,
                        ) = compute_all_gradient_metrics(
                            cos_model,
                            val_ldr,
                            test_ldr,
                            self.test_set,
                            self.train_set,
                            self.train_idxs[self.watch_train_client_id],
                            self.train_idxs[self.watch_val_client_id],
                            model_grads,
                            self.lr,
                            self.device,
                            self.dp,
                        )
                        save_dict["tarin_cos"] = train_cos
                        save_dict["val_cos"] = val_cos
                        save_dict["test_cos"] = test_cos
                        save_dict["tarin_diffs"] = train_diffs
                        save_dict["val_diffs"] = val_diffs
                        save_dict["test_diffs"] = test_diffs
                        save_dict["tarin_grad_norm"] = train_norm
                        save_dict["val_grad_norm"] = val_norm
                        save_dict["test_grad_norm"] = test_norm
                    if not os.path.exists(os.path.join(os.getcwd(), self.save_dir)):
                        os.makedirs(os.path.join(os.getcwd(), self.save_dir))
                        print(
                            "MIA Score saved in:",
                            os.path.join(os.getcwd(), self.save_dir),
                        )
                    torch.save(
                        save_dict,
                        os.path.join(
                            os.getcwd(),
                            self.save_dir,
                            f"client_{idx}_losses_epoch{epoch+1}.pkl",
                        ),
                    )

            if self.lr_up == "common":
                self.lr = self.lr * 0.99
            elif self.lr_up == "milestone":
                if epoch in self.schedule_milestone:
                    self.lr *= 0.1
            else:
                self.lr = lr_0 * (1 + math.cos(math.pi * epoch / self.epochs)) / 2

            client_weights = []
            for i in range(self.num_users):
                client_weight = len(
                    DatasetSplit(self.train_set, self.dict_users[i])
                ) / len(self.train_set)
                client_weights.append(client_weight)

            self._fed_avg(local_ws, client_weights, 1)
            self.model.load_state_dict(self.w_t)
            end = time.time()
            interval_time = end - start
            total_time += interval_time

            if (epoch + 1) == self.epochs or (epoch + 1) % 1 == 0:
                loss_train_mean, acc_train_mean = self.trainer.test(train_ldr)
                loss_val_mean, acc_val_mean = self.trainer.test(val_ldr)
                loss_test_mean, acc_test_mean = loss_val_mean, acc_val_mean

                self.logs["train_acc"].append(acc_train_mean)
                self.logs["train_loss"].append(loss_train_mean)
                self.logs["val_acc"].append(acc_val_mean)
                self.logs["val_loss"].append(loss_val_mean)
                self.logs["local_loss"].append(np.mean(local_losses))

                # use validation set as test set
                if self.logs["best_test_acc"] < acc_val_mean:
                    self.logs["best_test_acc"] = acc_val_mean
                    self.logs["best_test_loss"] = loss_val_mean
                    self.logs["best_model"] = copy.deepcopy(self.model.state_dict())

                print(
                    "Epoch {}/{}  --time {:.1f}".format(
                        epoch, self.epochs, interval_time
                    )
                )

                print(
                    "Train Loss {:.4f} --- Val Loss {:.4f}".format(
                        loss_train_mean, loss_val_mean
                    )
                )
                print(
                    "Train acc {:.4f} --- Val acc {:.4f} --Best acc {:.4f}".format(
                        acc_train_mean, acc_val_mean, self.logs["best_test_acc"]
                    )
                )
                s = "epoch:{}, lr:{:.5f}, val_acc:{:.4f}, val_loss:{:.4f}, tarin_acc:{:.4f}, train_loss:{:.4f},time:{:.4f}, total_time:{:.4f}".format(
                    epoch,
                    self.lr,
                    acc_val_mean,
                    loss_val_mean,
                    acc_train_mean,
                    loss_train_mean,
                    interval_time,
                    total_time,
                )

                with open(fn, "a") as f:
                    json.dump(
                        {
                            "epoch": epoch,
                            "lr": round(self.lr, 5),
                            "train_acc": round(acc_train_mean, 4),
                            "test_acc": round(acc_val_mean, 4),
                            "time": round(total_time, 2),
                        },
                        f,
                    )
                    f.write("\n")

        print(
            "------------------------------------------------------------------------"
        )
        print(
            "Test loss: {:.4f} --- Test acc: {:.4f}  ".format(
                self.logs["best_test_loss"], self.logs["best_test_acc"]
            )
        )

        return self.logs, interval_time, self.logs["best_test_acc"], acc_test_mean

    def _fed_avg(self, local_ws, client_weights, lr_outer):

        w_avg = copy.deepcopy(local_ws[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * client_weights[0]

            for i in range(1, len(local_ws)):
                w_avg[k] += local_ws[i][k] * client_weights[i]

            self.w_t[k] = w_avg[k]
