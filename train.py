
from preproc_utils import *
from model_utils import *
import pandas as pd
import torch
from torch import nn

if __name__ == '__main__':
    # PREPROCESSING
    #   ├ Import data and create the base dataset
    imgs_labels = pd.read_csv('data/images_labelling.csv')
    id_targets = dict(zip(imgs_labels['boxid'], imgs_labels['class_']))
    dataset = MyDataset(id_targets, imgs_folder='images/', transform=resize)
    unique_labels = np.unique(list(dataset.targets.values()))

    #   ├ Split data to train, test and validation parts
    split_sizes = (3000, 500, 250)
    splitted_data = train_test_val_split(dataset, sizes=split_sizes)

    #   ├ Filter team A and team B data
    a_team_splitted_data = [get_team_database(subset, 'A') for subset in splitted_data]
    b_team_splitted_data = [get_team_database(subset, 'B') for subset in splitted_data]

    #   └ Make batch iterators
    batch_size = 128
    dataloaders = dict(zip(['train', 'test', 'val'],
                           make_loaders(*splitted_data, batch_size=batch_size)))
    a_team_dataloaders = dict(zip(['train', 'test', 'val'],
                                  make_loaders(*a_team_splitted_data, batch_size=batch_size)))
    b_team_dataloaders = dict(zip(['train', 'test', 'val'],
                                  make_loaders(*b_team_splitted_data, batch_size=batch_size)))

    print('Preprocessing was done successfully!')

    # MODELS
    #   Scheme of work:
    #       preprocessed_data -> base_model ┬ (not team players) ────────────────┬──> output
    #                                       ├   (team A player)  -> a_team_model ┤
    #                                       └   (team B player)  -> b_team_model ┘

    calculate_loss = nn.CrossEntropyLoss()
    dump_dir = 'dumps/'

    # 1. Base model
    print('Training the base model...')
    #       ├ Define the model
    base_labels = ['A', 'B', 'C', 'D', 'other']
    base_model = initialize_efficientnet('efficientnet-b0', base_labels)

    #       ├ Train the model
    opt = torch.optim.AdamW(base_model.parameters(), lr=1e-4)
    train(base_model, 45, dump_dir + 'base.pth',
          base_labels, dataloaders, calculate_loss, opt,
          visualize_every=1, plot=False)

    #       └ Validate the model
    validation_results(base_model, base_labels, dataloaders['val'])

    # 2. Team A model
    print('Training the team A model...')
    #       ├ Define the model
    a_team_labels = unique_labels[isin_labels('A', unique_labels)]
    a_team_model = initialize_efficientnet('efficientnet-b3', a_team_labels)

    #       ├ Train the model
    opt = torch.optim.AdamW(base_model.parameters(), lr=1e-4)
    train(a_team_model, 45, dump_dir + 'a_team.pth',
          a_team_labels, a_team_dataloaders, calculate_loss, opt,
          visualize_every=1, plot=False)

    #       └ Validate the model
    validation_results(a_team_model, a_team_labels, a_team_dataloaders['val'])

    # 3. Team B model
    print('Training the team B model...')
    #       ├ Define the model
    b_team_labels = unique_labels[isin_labels('B', unique_labels)]
    b_team_model = initialize_efficientnet('efficientnet-b3', b_team_labels)

    #       ├ Train the model
    opt = torch.optim.AdamW(b_team_model.parameters(), lr=1e-4)
    train(b_team_model, 45, dump_dir + 'b_team.pth',
          a_team_labels, a_team_dataloaders, calculate_loss, opt,
          visualize_every=1, plot=False)

    #       └ Validate the model
    validation_results(b_team_model, b_team_labels, b_team_dataloaders['val'])

    print('The training is done. Best models are saved in the {} directory.'.format(dump_dir))
