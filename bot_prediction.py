
import torch
from preproc_utils import resize

# models initialization
models_dir = 'models/bot_models/'
base_model = torch.load(models_dir + 'base.pth', map_location=torch.device('cpu'))
a_team_model = torch.load(models_dir + 'ufa.pth', map_location=torch.device('cpu'))
b_team_model = torch.load(models_dir + 'dinamo.pth', map_location=torch.device('cpu'))

# all possible labels
labels = ['A11', 'A15', 'A19', 'A20', 'A31', 'A33', 'A5', 'A55', 'A57', 'A8',
          'A98', 'B1', 'B10', 'B22', 'B25', 'B27', 'B34', 'B4', 'B44', 'B48',
          'B5', 'B8', 'C-', 'D-', 'other']

# labels for base model. Every element should be in the elements of labels
team_model_labels = ['A', 'B', 'C', 'D', 'other']


def classify_img(img, base_model=base_model, a_team_model=a_team_model, b_team_model=b_team_model, aug=resize):
    """
    Classify the augmented image using the saved models
    :type img: torch.tensor
    :param img: input image, shape = [c x h x w]
    :param base_model: model wich makes base classification (teams, judges, others)
    :param b_team_model: model which classify players from team A
    :param a_team_model: model which classify players from team B
    :param aug: augmentation to be applied before the prediction
    :return:
    """
    img = aug(img).unsqueeze(0)
    team_pred = base_model(img).softmax(1)
    a_team_pred = a_team_model(img).softmax(1) * team_pred[:, 0]
    b_team_pred = b_team_model(img).softmax(1) * team_pred[:, 1]
    labels_prob = torch.cat([a_team_pred, b_team_pred, team_pred[:, 2:]], dim=1)
    prediction_dict = dict(zip(labels, labels_prob.flatten().tolist()))
    return prediction_dict
