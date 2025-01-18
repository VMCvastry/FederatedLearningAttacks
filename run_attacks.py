import sys
import os
from attacks.attack import attack_comparison


def main(argv):
    attack_modes = ["cosine attack", "grad diff", "loss based"]
    epochs = list(range(10, int(argv[2]) + 1, 10))
    p_folder = argv[1]
    PATH = argv[1]
    device = argv[3]

    MAX_K = 10
    FPR_THRESHOLD = "0.01"
    print(
        f"Starting with params: {p_folder}, {epochs}, {device}, {MAX_K}, {attack_modes}, {PATH}, {FPR_THRESHOLD}"
    )
    for root, dirs, files in os.walk(p_folder, topdown=False):
        print(f"Root: {root}, Directories: {dirs}")
        for name in dirs:
            print("names:", name)
            if len(name.split("_")) < 7 or len(name.split("_")[-1]) > 5:
                # print("【Error】:",os.path.join(root, name))
                continue
            elif root == p_folder:
                print(os.path.join(root, name))
                PATH = os.path.join(root, name)
                PATH += "/client_{}_losses_epoch{}.pkl"
                MAX_K = int(name.split("_K")[1].split("_")[0])
                model = name.split("_")[3]
                defence = name.split("_")[-5].strip("def").strip("0.0")
                seed = name.split("_")[-1]
                print("name:", name)
                log_path = "logs/log_alex"

                attack_comparison(
                    PATH,
                    log_path,
                    epochs,
                    MAX_K,
                    defence,
                    seed,
                    attack_modes,
                    FPR_THRESHOLD,
                )
                print("success!")

            print(os.path.join(root, name))


if __name__ == "__main__":
    main(sys.argv)
