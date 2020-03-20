import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from preproc_utils import isin_labels
from IPython.display import clear_output
from sklearn.metrics import classification_report
from efficientnet_pytorch import EfficientNet


def keyword_ohe(labels, label_classes):
    """
    Performs one hot encoding of the specified classes from the labels
    :param labels: initial sequence of labels
    :param label_classes: set of required classes from labels
    :return: encoded matrix, shape: [len(label_classes) x len(labels)]
    """
    encoded_labels = []
    for i, l in enumerate(label_classes):
        class_in_label_np = isin_labels(l, np.array(labels))
        class_in_label_torch = torch.tensor(class_in_label_np, dtype=torch.uint8)
        encoded_labels.append(class_in_label_torch.unsqueeze(1))
    return torch.cat(encoded_labels, dim=1)


ohe_to_int = lambda ohe: torch.argmax(ohe, 1)  # convert ohe matrix to sequence of classes


def initialize_efficientnet(model_name, model_labels):
    """
    Efficientnet model initialization: load pretrained + reshape out dense layer
    :param model_name: efficientnet structure name
    :param model_labels: set of possible labels model should predict (used for reshaping dense layer)
    :return:
    """
    model = EfficientNet.from_pretrained(model_name)
    model._fc = nn.Linear(model._fc.in_features, len(model_labels))
    if torch.cuda.is_available():
        model.cuda()
    return model


def train_an_epoch(model, required_labels, dataloaders, calculate_loss, opt):
    """
    One-epoch model trainer
    :param model: torch model to train
    :param required_labels: set of labels to predict
    :param dataloaders: dictionary of batch iterators, 'Train' key in the dictionary is necessary
    :param calculate_loss: function f: f(pred, target) -> loss
    :param opt: optimizer
    :return: losses
    """
    epoch_losses = dict()
    for mode in dataloaders.keys():

        if mode == 'train':
            model.train()
        elif mode == 'test':
            model.eval()
        elif mode == 'val':
            continue

        losses = []
        for imgs, labels in dataloaders[mode]:
            target = ohe_to_int(keyword_ohe(labels, required_labels))

            if torch.cuda.is_available():
                imgs, target = imgs.cuda(), target.cuda()

            pred = model(imgs)
            loss = calculate_loss(pred, target)

            losses.append(loss.item())

            if mode == 'train':
                loss.backward()
                opt.step()
                opt.zero_grad()

        epoch_losses[mode] = np.mean(losses)
    return epoch_losses


def visualize_loss(losses_dict, visualize_every=1, plot=True):
    """
    Function which visualize training process
    :type losses_dict: dict
    :type visualize_every: int
    :type plot: bool
    :param losses_dict: dictionary of losses to visualize
    :param visualize_every: epoch step to visualize
    :param plot: flag of showing training plot, if False function prints losses
    """
    epoch_i = len(list(losses_dict.values())[0])
    if epoch_i % visualize_every == 0:
        clear_output(True)

        if plot:
            plt.figure(figsize=(25, 8))
            for key, loss in losses_dict.items():
                plt.plot(loss, label='{} loss: {:.4f}'.format(key, loss[-1]))
                plt.legend()
            plt.show()
        else:
            print('Epoch #{}'.format(epoch_i))
            for key, loss in losses_dict.items():
                print('\t{} loss: {:.4f}'.format(key, loss[-1]))


def train(model, n_epochs, dump_file, *train_args, **visualizer_kwargs):
    """
    Train the model during n_epochs with constant learning rate and dumping best model.
    :param model: torch model to train
    :param n_epochs: number of epochs to train
    :param dump_file: file to dump the best model
    :param train_args: should contain [required_labels, dataloaders, calculate_loss, opt]
    (see docstring for train_an_epoch)
    :param visualizer_kwargs: flags for training visualizer (see docstring for visualize_loss)
    """
    epoch_i = 0

    losses_dict = {'train': [], 'test': []}
    while epoch_i <= n_epochs:
        epoch_losses = train_an_epoch(model, *train_args)
        for mode in losses_dict.keys():
            losses_dict[mode].append(epoch_losses[mode])

        epoch_i += 1

        visualize_loss(losses_dict, **visualizer_kwargs)

        if dump_file and epoch_i > 2:
            if ((epoch_losses['train'] < min(losses_dict['train'][:-1]))
                    and (epoch_losses['test'] < min(losses_dict['test'][:-1]))):
                print('Model saved => {}'.format(dump_file))
                torch.save(model, dump_file)


def validation_results(model, required_labels, val_dataloader):
    """
    Prints results of model validation: cross entropy loss and classification error
    :param model: model to validate
    :param required_labels: labels which model predicts
    :param val_dataloader: validation batches iterator
    """
    model.cpu()
    model.eval()
    ce_loss = nn.CrossEntropyLoss()

    losses = []
    for imgs, labels in val_dataloader:
        target = ohe_to_int(keyword_ohe(labels, required_labels))
        imgs, target = imgs.cpu(), target.cpu()

        pred = model(imgs)
        loss = ce_loss(pred, target)
        losses.append(loss.item())

        pred = pred.softmax(1).argmax(1)

    print('Validation:\n\tCross Entropy: {:.4f}'.format(np.mean(losses)))
    print('\tClassification report:', '_' * 22)
    print(classification_report(target.cpu().detach(), pred.cpu().detach(), target_names=required_labels))
