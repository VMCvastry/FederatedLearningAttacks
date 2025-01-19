from typing import List
import numpy as np
import torch
import scipy
from .attack_utils import *

import warnings

warnings.filterwarnings("ignore")
MODE = "val"


def cos_attack_single(
    f,
    K,
    epch,
    attack_mode="loss based",
    extract_fn=None,
):
    """
    A single-model membership inference that uses either:
      - "cosine attack" (cos),
      - "grad diff" (diff),
      - "loss based" (CE).

    This effectively subsumes 'common_attack' if you choose attack_mode="loss based".

    :param f: Format string for loading "f.format(0, epch)"
    :param K: Not used here except for signature consistency
    :param epch: Current epoch
    :param attack_mode: "cosine attack", "grad diff", or "loss based"
    :param mode: "test" or "val" to pick which set is considered non-member
    :return: (accs, tprs, auc, log_auc, (val_liratios, train_liratios))
    """
    if extract_fn is not None:
        raise ValueError("extract_fn is not used in this function")

    accs = []
    # Load only the target client 0
    target_res = torch.load(f.format(0, epch))
    accs.append(target_res.get("test_acc", 0.0))
    tprs = None

    # Depending on the chosen mode, pick the relevant arrays from target_res
    # For the "non-member" side
    if attack_mode == "cosine attack":
        if MODE == "test":
            val_liratios = target_res["test_cos"]
        else:  # "val"
            val_liratios = target_res["val_cos"]
        val_liratios = -np.array([v.cpu().item() for v in val_liratios])

        train_liratios = target_res["tarin_cos"]
        train_liratios = -np.array([v.cpu().item() for v in train_liratios])

        auc_val, log_auc_val, tprs_val = calc_auc(
            "cos_attack", torch.tensor(val_liratios), torch.tensor(train_liratios), epch
        )
        return accs, tprs_val, auc_val, log_auc_val, (val_liratios, train_liratios)

    elif attack_mode == "grad diff":
        if MODE == "test":
            val_liratios = target_res["test_diffs"]
        else:  # "val"
            val_liratios = target_res["val_diffs"]
        val_liratios = np.array([v.cpu().item() for v in val_liratios])

        train_liratios = target_res["tarin_diffs"]
        train_liratios = np.array([v.cpu().item() for v in train_liratios])

        auc_val, log_auc_val, tprs_val = calc_auc(
            "diff_attack",
            torch.tensor(val_liratios),
            torch.tensor(train_liratios),
            epch,
        )
        return accs, tprs_val, auc_val, log_auc_val, (val_liratios, train_liratios)

    elif attack_mode == "loss based":
        # This is exactly what 'common_attack' was doing
        if MODE == "test":
            val_liratios = -ce_loss_fn(
                target_res["test_res"]["logit"], target_res["test_res"]["labels"]
            )
        else:  # "val"
            val_liratios = -ce_loss_fn(
                target_res["val_res"]["logit"], target_res["val_res"]["labels"]
            )
        train_liratios = -ce_loss_fn(
            target_res["train_res"]["logit"], target_res["train_res"]["labels"]
        )

        auc_val, log_auc_val, tprs_val = calc_auc(
            "loss_attack",
            torch.tensor(val_liratios),
            torch.tensor(train_liratios),
            epch,
        )
        return accs, tprs_val, auc_val, log_auc_val, (val_liratios, train_liratios)

    else:
        raise ValueError(f"Unknown attack_mode: {attack_mode}")


def lira_attack_ldh_cosine(f, epch, K, extract_fn=None, attack_mode="cos"):
    """
    A multi-client approach that uses gradient cos or diff features
    from each of the K clients to form a shadow distribution.
    This is more advanced and corresponds to the LIRA approach
    with either 'cos' or 'diff'.
    """
    accs = []
    training_res = []
    for i in range(K):
        training_res.append(torch.load(f.format(i, epch)))
        accs.append(training_res[-1].get("test_acc", 0.0))

    # Target is client 0
    target_res = training_res[0]
    shadow_res = training_res[1:]

    # Extract target features
    if attack_mode == "cos":
        # (tarin_cos, test_cos, val_cos, etc.)
        target_train_loss = torch.tensor(target_res["tarin_cos"]).cpu().numpy()
        if MODE == "test":
            target_test_loss = torch.tensor(target_res["test_cos"]).cpu().numpy()
        else:  # "val"
            target_test_loss = torch.tensor(target_res["val_cos"]).cpu().numpy()
    elif attack_mode == "diff":
        target_train_loss = torch.tensor(target_res["tarin_diffs"]).cpu().numpy()
        if MODE == "test":
            target_test_loss = torch.tensor(target_res["test_diffs"]).cpu().numpy()
        else:  # "val"
            target_test_loss = torch.tensor(target_res["val_diffs"]).cpu().numpy()
    else:
        raise ValueError(f"Unknown attack_mode: {attack_mode}")

    # Build shadow distribution
    shadow_train_losses = []
    shadow_test_losses = []
    for sres in shadow_res:
        if attack_mode == "cos":
            shadow_train_losses.append(torch.tensor(sres["tarin_cos"]).cpu().numpy())
            if MODE == "val":
                shadow_test_losses.append(torch.tensor(sres["val_cos"]).cpu().numpy())
            else:
                shadow_test_losses.append(torch.tensor(sres["test_cos"]).cpu().numpy())
        else:  # "diff"
            shadow_train_losses.append(torch.tensor(sres["tarin_diffs"]).cpu().numpy())
            if MODE == "val":
                shadow_test_losses.append(torch.tensor(sres["val_diffs"]).cpu().numpy())
            else:
                shadow_test_losses.append(
                    torch.tensor(sres["test_diffs"]).cpu().numpy()
                )

    shadow_train_losses = np.vstack(shadow_train_losses)
    shadow_test_losses = np.vstack(shadow_test_losses)

    # Fit normal distribution
    train_mu_out = shadow_train_losses.mean(axis=0)
    train_var_out = shadow_train_losses.var(axis=0) + 1e-8
    test_mu_out = shadow_test_losses.mean(axis=0)
    test_var_out = shadow_test_losses.var(axis=0) + 1e-8

    # membership: train is member, test is non-member => invert cdf
    # 1 - cdf(...) => bigger => more likely member
    train_l_out = 1 - scipy.stats.norm.cdf(
        target_train_loss, train_mu_out, np.sqrt(train_var_out)
    )
    test_l_out = 1 - scipy.stats.norm.cdf(
        target_test_loss, test_mu_out, np.sqrt(test_var_out)
    )

    auc_val, log_auc_val, tprs_val = calc_auc(
        "lira", torch.tensor(test_l_out), torch.tensor(train_l_out), epch
    )
    # Return (accs, tprs, auc, log_auc, membership_scores)
    return accs, tprs_val, auc_val, log_auc_val, (train_l_out, test_l_out)


@torch.no_grad()
def attack_comparison(
    p: str,
    log_path: str,
    epochs: List[int],
    MAX_K: int,
    defence: str,
    seed: str,
    attack_modes: List[str],
    fpr_threshold: float,
):
    """
    Orchestrates the membership inference attacks across multiple epochs,
    computes metrics, and plots the results.

    Parameters:
    -----------
    p        : str
        Path template for loading .pkl files. Example: "path/to/client_{}_epoch{}.pkl".
    log_path : str
        Directory path for saving logs/plots.
    epochs   : List[int]
        A list of epochs at which membership inference is evaluated.
    MAX_K    : int
        Number of clients (federated setting).
    defence  : str
        Name of the defense method used (if any).
    seed     : str
        Random seed or run identifier.

    Returns:
    --------
    None
        Saves a PDF plot and a .log file with TPR metrics and other data.
    """

    # 1. Run LIRA Attack on the last epoch to get final accuracy
    #    ------------------------------------------------------
    try:
        final_lira_result = lira_attack_ldh_cosine(
            p, epochs[-1], K=MAX_K, extract_fn=extract_hinge_loss
        )
        # final_lira_result usually returns something like:
        # (accs, tprs, auc, log_auc, (train_scores, test_scores))
        final_acc = final_lira_result[0]  # The test accuracy array
    except (ValueError, IndexError) as e:
        print(f"Error loading LIRA on final epoch {epochs[-1]}: {e}")
        return  # Cannot proceed if we fail here

    # 2. Prepare data structures to track results
    #    ----------------------------------------
    # reses_*: Will store arrays of membership scores to be averaged later
    reses_lira = []
    reses_common = {
        k: [] for k in attack_modes
    }  # e.g. "cosine attack", "grad diff", etc.

    # *scores* holds TPR@FPR=FPR_THRESHOLD for each epoch
    scores = {k: [] for k in attack_modes}
    scores["lira"] = []  # we'll store TPR@FPR=FPR_THRESHOLD for LIRA as well

    # *avg_scores* will store the final aggregated TPR dictionary for each attack
    avg_scores = {k: [] for k in attack_modes}
    avg_scores["lira"] = []

    # Some lists for quick logging or checks
    lira_scores = []
    common_scores = []

    # *other_scores* collects optional debug info or single-epoch results for special cases
    other_scores = {}

    print("\n=== Starting Attack Comparison ===")
    print(f"Epochs to evaluate: {epochs}")
    print(f"Using up to {MAX_K} clients, defense: {defence}, seed: {seed}\n")

    # 3. Main loop: run LIRA + 'common' attacks for each epoch
    #    -----------------------------------------------------
    for epch in epochs:
        print(f"Evaluating epoch={epch}")

        try:
            # LIRA: lira_attack_ldh_cosine
            lira_score = lira_attack_ldh_cosine(
                p, epch, K=MAX_K, extract_fn=extract_hinge_loss
            )
        except ValueError:
            # Possibly a file-loading or shape mismatch
            print(f"ValueError at epoch={epch}, skipping...")
            continue

        # lira_score: (accs, tprs, auc, log_auc, (train_scores, test_scores))
        # We record TPR@FPR=FPR_THRESHOLD from the "tprs" dictionary
        lira_tprs = lira_score[1]
        lira_tpr_fpr01 = lira_tprs.get(fpr_threshold, 0.0)

        scores["lira"].append(lira_tpr_fpr01)  # store per-epoch TPR@FPR=FPR_THRESHOLD
        lira_scores.append(lira_tpr_fpr01)
        reses_lira.append(lira_score[-1])  # the raw (train_scores, test_scores)

        # Calc running average
        train_arrays = [pair[0].reshape(1, -1) for pair in reses_lira]
        test_arrays = [pair[1].reshape(1, -1) for pair in reses_lira]

        train_score = np.vstack(train_arrays).mean(axis=0)
        test_score = np.vstack(test_arrays).mean(axis=0)

        auc_val, log_auc_val, tprs_val = calc_auc(
            "averaged_lira_running",
            torch.tensor(test_score),
            torch.tensor(train_score),
            epch,
        )

        fpr_str = str(fpr_threshold)  #
        avg_scores["lira"].append(tprs_val.get(fpr_str, 0.0))

        # Run 'common' attacks (e.g. "cosine attack", "grad diff", "loss based"):
        for attack_mode in attack_modes:
            common_score = cos_attack_single(
                p,
                0,
                epch,
                attack_mode,
            )  # extract_fn=extract_hinge_loss
            # common_score: (accs, tprs, auc, log_auc, (val_liratios, train_liratios))

            # Store the TPR@FPR=FPR_THRESHOLD
            tpr_fpr01 = common_score[1].get(fpr_threshold, 0.0)
            scores[attack_mode].append(tpr_fpr01)
            common_scores.append(tpr_fpr01)

            # The raw membership scores for aggregator
            reses_common[attack_mode].append(common_score[-1])

            # Calc running average
            train_arrays = [
                x[0].reshape(1, -1) for x in reses_common[attack_mode]
            ]  # x[0] => val_liratios
            val_arrays = [
                x[1].reshape(1, -1) for x in reses_common[attack_mode]
            ]  # x[1] => train_liratios

            mean_val = -np.vstack(val_arrays).mean(axis=0)
            mean_train = -np.vstack(train_arrays).mean(axis=0)

            auc_, log_auc_, tprs_ = calc_auc(
                "avg_" + attack_mode,
                torch.tensor(mean_val),  # first => "non-member" side
                torch.tensor(mean_train),  # second => "member" side
                epch,
            )

            avg_scores[attack_mode].append(tprs_.get(fpr_str, 0.0))

            # If epoch=200 and "loss based", store additional info
            if epch == 200 and attack_mode == "loss based":
                other_scores["loss_single_epch_score"] = common_score[
                    1
                ]  # entire TPR dict
                other_scores["loss_single_auc"] = [
                    common_score[2],
                    common_score[3],
                ]  # (auc, log_auc)

    # 6. Show final TPR@FPR=FPR_THRESHOLD for each epoch & each attack
    #    ----------------------------------------------------
    print("=== Per-epoch TPR@FPR=FPR_THRESHOLD scores ===")
    for attack_mode, tpr_values in scores.items():
        print(
            f" Attack: {attack_mode}, TPR@FPR=FPR_THRESHOLD across epochs: {tpr_values}"
        )

    # 7. Save figure & logs via fig_out
    #    --------------------------------
    fig_out(
        x_axis_data=epochs,
        MAX_K=MAX_K,
        defence=defence,
        seed=seed,
        log_path=log_path,
        d=scores,
        avg_d=avg_scores,
        other_scores=other_scores,
        accs=final_acc,
        fpr_val=fpr_threshold,
    )
    print("Plot and log have been saved.\n")


# --------------------------------------------------------------------------
# Helper function: Filter infinite/nan values in an array
def _filter_inf_nan(arr: np.ndarray, fill_value: float = -1e10) -> np.ndarray:
    arr = arr[~np.isnan(arr)]  # remove NaNs
    arr[np.isinf(arr)] = fill_value  # replace +/- inf
    return arr
