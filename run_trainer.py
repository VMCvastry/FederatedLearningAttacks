import os
from training.global_trainer import FederatedLearning

from utils.args import parser_args
import random
import numpy as np
import torch

import models as models


def main(args):
    logs = {
        "net_info": None,
        "arguments": {
            "frac": args.frac,
            "local_ep": args.local_ep,
            "local_bs": args.batch_size,
            "lr_outer": args.lr_outer,
            "lr_inner": args.lr,
            "iid": args.iid,
            "wd": args.wd,
            "optim": "sgd",
            "model_name": args.model_name,
            "dataset": args.dataset,
            "log_interval": args.log_interval,
            "num_classes": args.num_classes,
            "epochs": args.epochs,
            "num_users": args.num_users,
        },
    }
    save_dir = args.save_dir
    fl = FederatedLearning(args)

    logg, time, best_test_acc, test_acc = fl.train()

    logs["net_info"] = logg
    logs["test_acc"] = test_acc
    logs["bp_local"] = True if args.bp_interval == 0 else False

    if not os.path.exists(save_dir + args.model_name + "/" + args.dataset):
        os.makedirs(save_dir + args.model_name + "/" + args.dataset)
    torch.save(
        logs,
        save_dir
        + args.model_name
        + "/"
        + args.dataset
        + "/epoch_{}_E_{}_u_{}_{:.4f}_{:.4f}.pkl".format(
            args.epochs, args.local_ep, args.num_users, time, test_acc
        ),
    )
    return


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    args = parser_args()
    print(args)
    setup_seed(args.seed)

    args.save_dir = (
        args.save_dir
        + "/"
        + f"{args.dataset}_K{args.num_users}_N{args.samples_per_user}_{args.model_name}_def{args.defense}_iid_sgd_local{args.local_ep}_s{args.seed}"
    )
    print("scores saved in:", os.path.join(os.getcwd(), args.save_dir))
    args.log_folder_name = args.save_dir
    main(args)
