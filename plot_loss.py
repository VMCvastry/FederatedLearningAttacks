import json
import matplotlib.pyplot as plt
import sys
import os


def read_log_file(file_path):
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]
    return data


def plot_accuracies(data, log_file_path):
    epochs = [entry["epoch"] for entry in data]
    train_acc = [entry["train_acc"] for entry in data]
    test_acc = [entry["test_acc"] for entry in data]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
    plt.plot(epochs, test_acc, label="Test Accuracy", marker="o")

    plt.title("Train and Test Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    log_filename = os.path.splitext(os.path.basename(log_file_path))[0]
    save_dir = os.path.join("logs", "train_plots")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{log_filename}.png")
    plt.savefig(save_path)
    print(f"Plot saved at: {save_path}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <log_file_path>")
        sys.exit(1)

    log_file_path = sys.argv[1]

    log_data = read_log_file(log_file_path)

    plot_accuracies(log_data, log_file_path)

# python3 plot_loss.py "../FedMIA/log_fedmia_4_300/log_fedmia_4_300/cifar100_K10_N5000_alexnet_defnone_iid_\$sgd_local4_s1/a_alexnet_cifar100_10_sgd_cosine_100_2025_01_04_162500.log"
# python3 plot_loss.py "../FedMIA/log_fedmia/cifar100_K10_N5000_alexnet_defnone_iid_\$sgd_local1_s1/a_alexnet_cifar100_10_sgd_cosine_100_2025_01_02_233151.log"
# python3 plot_loss.py "../FedMIA/log_fedmia/cifar100_K10_N5000_alexnet_defnone_iid_\$sgd_local1_s1/a_alexnet_cifar100_10_sgd_cosine_100_2025_01_02_172418.log"
