import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the top-k predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TrainerPrivate(object):
    def __init__(self, model, train_loader, device, sigma, num_classes, batch_size=100):
        self.model = model
        self.device = device
        self.sigma = sigma
        self.num_classes = num_classes
        self.train_loader = train_loader
        self.batch_size = batch_size

    def _initialize_optimizer(self, lr):
        """Initialize the optimizer."""
        return optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
        )

    def local_update_noback(self, dataloader, local_epochs, lr):
        """
        Perform local model update without backward propagation.
        """
        optimizer = self._initialize_optimizer(lr)
        epoch_losses = []

        for _ in range(local_epochs):
            avg_loss, _ = self._train_one_epoch(dataloader, optimizer)
            epoch_losses.append(avg_loss)

        return self.model.state_dict(), np.mean(epoch_losses)

    def _train_one_epoch(self, dataloader, optimizer):
        """Train the model for one epoch."""
        # self.model.train()
        total_loss = 0
        total_accuracy = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(x)
            loss = F.cross_entropy(predictions, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_accuracy += accuracy(predictions, y)[0].item()

        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        return avg_loss, avg_accuracy

    def test(self, dataloader):

        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0
        runcount = 0

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)

                pred = self.model(data)  # test = 4
                loss_meter += F.cross_entropy(
                    pred, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = pred.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0)

        loss_meter /= runcount
        acc_meter /= runcount

        return loss_meter, acc_meter
