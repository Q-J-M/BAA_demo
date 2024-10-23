from utils import *
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def validate(test_image, test_label, test_gender, model, epoch):
    train_data, train_ref, train_gender = test_image, test_label, test_gender

    model.eval()

    mse = 0

    with torch.no_grad():

        train_image = to_var(train_data).detach()
        train_gender = to_var(train_gender.view(-1, 1)).detach()
        train_age = to_var(train_ref.view(-1, 1)).detach()

        out = model(train_image, train_gender)

        loss = F.mse_loss(out, train_age)
        val_loss = loss.detach().cpu().item()


        with open('val_loss.txt', 'a') as f:
            f.write(str(epoch) + ',' + str(val_loss) + '\n')

        predictions = out.detach().cpu().numpy()
        true_labels = train_age.detach().cpu().numpy()

        mse = mean_squared_error(true_labels, predictions)
        mae = mean_absolute_error(true_labels, predictions)
        r2 = r2_score(true_labels, predictions)




    return mse,mae





