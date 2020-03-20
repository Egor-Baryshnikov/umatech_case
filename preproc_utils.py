from torch.utils import data as tdata
from torchvision import transforms as tt
from skimage import io
import copy
import numpy as np


class MyDataset(tdata.Dataset):
    """
    Dataset which takes id-target dict and loads the corresponding images.
    Names of the images should be '<id>.png'.
    """

    def __init__(self, targets, imgs_folder, transform=None):
        """
        :param targets: id-target label dictionary
        :param imgs_folder: folder with the images
        :param transform: transforms of the returned X
        """
        self.imgs_folder = imgs_folder
        self.ids = list(targets.keys())
        self.img_names = map(lambda id_: self.imgs_folder + str(id_) + '.png', self.ids)
        self.imgs = map(lambda file: io.imread(file), self.img_names)
        self.data = dict(zip(self.ids, list(self.imgs)))
        self.targets = targets

        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.data[self.ids[index]]
        y = self.targets[self.ids[index]]

        if self.transform:
            x = self.transform(x)

        return x, y


class DatasetFromSubset(tdata.Dataset):
    """
    Subset which defined image transformations
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def train_test_val_split(data, sizes):
    """
    Splits data to train, test and validation subsets.
    :param data: Dataset to be split
    :param sizes: lengths of splits to be produced
    :return: train subset, test subset, val subset
    """
    train_data, test_data, val_data = tdata.random_split(data, sizes)

    train_data = DatasetFromSubset(train_data, transform=augm)
    test_data = DatasetFromSubset(test_data, transform=augm)
    val_data = DatasetFromSubset(val_data, transform=resize)
    return train_data, test_data, val_data


def make_loaders(train_data, test_data, val_data, batch_size, n_workers=16):
    """
    Creates DataLoaders for train, test, val subsets.
    :param train_data: train subset
    :param test_data: test subset
    :param val_data: validation subset
    :param batch_size: length of the batch which will be returned from DataLoader
    :param n_workers: how many subprocesses to use for data loading
    :return: train dataloader, test dataloader, val dataloader
    """
    train_dataloader = tdata.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    test_dataloader = tdata.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_dataloader = tdata.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    return train_dataloader, test_dataloader, val_dataloader


isin_labels = lambda str_, labels: np.vectorize(lambda x: str_ in x)(labels)  # check if letter-indicator is in labels


def get_team_database(data, team):
    """
    Forms dataset with specified team from data
    :type team: str
    :type data: DatasetFromSubset
    :param data: Subset with initial data
    :param team: Letter-indicator of the needed team in the initial data target
    :return: team dataset
    """
    if not isinstance(data, DatasetFromSubset):
        raise ExeptionError
    global_targets = list(data.subset.dataset.targets.values())
    subset_indices = np.array(data.subset.indices)
    subset_targets = np.array(global_targets)[subset_indices]
    isin_team = isin_labels(team, subset_targets)
    team_indices = np.where(isin_team == True)[0]
    data_team = copy.deepcopy(data)
    data_team.subset.indices = subset_indices[team_indices]
    return data_team

# AUGMENTATIONS
#       - Resize to default shape
resize = tt.Compose([
    tt.ToPILImage(),
    tt.Resize(size=(100, 50)),
    tt.ToTensor()
])

#       - Convert image to tensor
totensor = tt.ToTensor()

#       - Train augmentation: resize and small rotation.
#             Note:
#             Because of we classify images from one video translation,
#             it is not necessary to add augmentation with colors.
#             Moreover, bounding box images contains object with relatively same size,
#             So the cropping is not critical.
augm = tt.Compose([
    tt.ToPILImage(),
    tt.Resize(size=(100, 50)),
    tt.RandomRotation(10, fill=0),
    tt.ToTensor()
])
