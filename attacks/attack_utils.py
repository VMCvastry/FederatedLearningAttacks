import numpy as np
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
import scipy
import json


def liratio(mu_in, mu_out, var_in, var_out, new_samples):
    """Computes likelihood ratio using normal distribution CDF."""
    return scipy.stats.norm.cdf(new_samples, mu_out, np.sqrt(var_out))


@torch.no_grad()
def hinge_loss_fn(x, y):
    """Computes hinge loss for logits."""
    x, y = copy.deepcopy(x).cuda(), copy.deepcopy(y).cuda()
    mask = torch.eye(x.shape[1], device="cuda")[y].bool()
    tmp1 = x[mask]
    x[mask] = -1e10
    tmp2 = torch.max(x, dim=1)[0]
    return (tmp1 - tmp2).cpu().numpy()


def ce_loss_fn(x, y):
    """Computes cross-entropy loss."""
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    return loss_fn(x, y)


def merge_dicts(lst):
    """Merges a list of dictionaries into one."""
    new_dict = {}
    for d in lst:
        for k, v in d.items():
            if k in new_dict:
                new_dict[k].extend(v)
            else:
                new_dict[k] = v
    return new_dict


def get_logit(l):
    """Computes logits from losses."""
    return np.log(np.exp(-np.array(l)) / (1 - np.exp(-np.array(l)) + 1e-15))


def extract_hinge_loss(i):
    val_dict = {}
    val_index = i["val_index"]
    val_hinge_index = hinge_loss_fn(i["val_res"]["logit"], i["val_res"]["labels"])
    for j, k in zip(val_index, val_hinge_index):
        if j in val_dict:
            val_dict[j].append(k)
        else:
            val_dict[j] = [k]

    train_dict = {}
    train_index = i["train_index"]
    train_hinge_index = hinge_loss_fn(i["train_res"]["logit"], i["train_res"]["labels"])
    for j, k in zip(train_index, train_hinge_index):
        if j in train_dict:
            train_dict[j].append(k)
        else:
            train_dict[j] = [k]

    test_dict = {}
    test_index = i["test_index"]
    test_hinge_index = hinge_loss_fn(i["test_res"]["logit"], i["test_res"]["labels"])
    for j, k in zip(test_index, test_hinge_index):
        if j in test_dict:
            test_dict[j].append(k)
        else:
            test_dict[j] = [k]

    return (val_dict, train_dict, test_dict)


def extract_loss2logit(i):
    val_dict = {}
    val_index = i["val_index"]
    val_hinge_index = get_logit(i["val_res"]["loss"])
    for j, k in zip(val_index, val_hinge_index):
        if j in val_dict:
            val_dict[j].append(k)
        else:
            val_dict[j] = [k]
    train_dict = {}
    train_index = i["train_index"]
    train_hinge_index = get_logit(i["train_res"]["loss"])
    for j, k in zip(train_index, train_hinge_index):
        if j in train_dict:
            train_dict[j].append(k)
        else:
            train_dict[j] = [k]
    return (val_dict, train_dict)


def extract_loss(i):
    val_dict = {}
    for j, k in zip(i["val_index"], i["val_res"]["loss"]):
        if j in val_dict:
            # assert 0
            val_dict[j].append(k)
        else:
            val_dict[j] = [k]
    train_dict = {}
    for j, k in zip(i["train_index"], i["train_res"]["loss"]):
        if j in train_dict:
            train_dict[j].append(k)
        else:
            train_dict[j] = [k]
    return (val_dict, train_dict)


def calc_auc(name, target_val_score, target_train_score, epoch):
    global fpr, tpr
    fpr, tpr, thresholds = metrics.roc_curve(
        torch.cat(
            [torch.zeros_like(target_val_score), torch.ones_like(target_train_score)]
        )
        .cpu()
        .numpy(),
        torch.cat([target_val_score, target_train_score]).cpu().numpy(),
    )
    auc = metrics.auc(fpr, tpr)
    log_tpr, log_fpr = np.log10(tpr), np.log10(fpr)
    log_tpr[log_tpr < -5] = -5
    log_fpr[log_fpr < -5] = -5
    log_fpr = (log_fpr + 5) / 5.0
    log_tpr = (log_tpr + 5) / 5.0
    log_auc = metrics.auc(log_fpr, log_tpr)

    tprs = {}
    for fpr_thres in [0.1, 0.02, 0.01, 0.001, 0.0001]:
        tpr_index = np.sum(fpr < fpr_thres)
        tprs[str(fpr_thres)] = tpr[tpr_index - 1]
    return auc, log_auc, tprs


def fig_out(
    x_axis_data,
    MAX_K,
    defence,
    seed,
    log_path,
    d,
    avg_d=None,
    other_scores=None,
    accs=None,
    fpr_val="0.01",
):
    colors = {
        "cosine attack": "r",
        "loss based": "b",
        "lira": "y",
    }
    fig = plt.figure(figsize=(6.5, 6.5), dpi=200)
    fig.subplots_adjust(
        top=0.91, bottom=0.160, left=0.180, right=0.9, hspace=0.2, wspace=0.2
    )
    for k in d.keys():
        plt.plot(
            x_axis_data[0 : len(d[k])], d[k], linewidth=1, label=k, color=colors[k]
        )
    # plt.plot(x_axis_data, common_score,'bo-', linewidth=1, color='#2E8B57', label=r'Baseline')
    plt.legend(loc=3)

    plt.xlim(-2, 305)
    my_x_ticks = np.arange(0, 302, 50)
    plt.xticks(my_x_ticks, size=14)
    if avg_d:
        for k in avg_d.keys():
            if avg_d[k]:
                plt.plot(
                    x_axis_data[: len(avg_d[k])],
                    avg_d[k],
                    label="avg_" + k,
                    color=colors[k],
                    linestyle="--",
                )

    plt.legend(prop={"size": 10})
    plt.xlabel("Epoch", fontsize=14, fontdict={"size": 14})  # x_label
    plt.ylabel("TPR@FPR=" + fpr_val, fontsize=14, fontdict={"size": 14})  # y_label
    plt.grid(axis="both")

    pdf_path = (
        log_path
        + f"/def{defence}2_0.85_k{MAX_K}_{seed}_{len(x_axis_data*10)}_{fpr_val}_attack.png"
    )
    plt.savefig(pdf_path)

    print("log_path0:", log_path)
    log_path = (
        log_path
        + f"/def{defence}2_0.85_k{MAX_K}_{seed}_{len(x_axis_data*10)}_{fpr_val}_attack.log"
    )
    print("log_path:", log_path)
    with open(log_path, "w") as f:
        json.dump({"avg_d": avg_d, "other_scores": other_scores, "accs": accs}, f)
    # assert 0
