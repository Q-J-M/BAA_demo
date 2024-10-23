from dataloader import build_datasets
from models.simpleNet import SimpleNet
import args_parser
import torch.nn as nn
import torch
import os
from train import train
from torch.utils.data import DataLoader, TensorDataset
from validate import validate
from tqdm import tqdm
from models.Unet3D import UNet3D

args = args_parser.args_parse()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():

    with open('test_data.txt', 'w') as f:
        pass

    # dataloader
    train_list, test_list, train_gender, test_gender = build_datasets()



    dataset = TensorDataset(train_list[0], train_list[1], train_gender)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = TensorDataset(test_list[0], test_list[1], test_gender)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    if args.arch == 'simpleNet':
        model = SimpleNet(args.gender_num)
    if args.arch == 'Unet3D':
        model = UNet3D(1,1,args.gender_num)

    if torch.cuda.is_available():
        model = model.to('cuda')

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with open('train_loss.txt', 'w') as f:
        pass

    with open('val_loss.txt', 'w') as f:
        pass

    model_path = args.model_path.replace('arch', args.arch)

    best_mae = float('inf')


    print('Start Training: ')
    for epoch in range(args.n_epochs):
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.n_epochs}', unit='batch')
        for batch in train_loader_tqdm:
            image, label, gender  = batch
            train_loader_tqdm.set_postfix()
            # print('Train_Epoch_{}: '.format(epoch))
            train(image, label, gender,
                    model,
                  optimizer,
                  criterion,
                  epoch,
                  )

        # print('Val_Epoch_{}: '.format(epoch))
        sum_mse = 0
        sum_mae = 0

        for batch in test_loader:
            test_image, test_label, test_gender = batch
            recent_mse,recent_mae = validate(test_image, test_label, test_gender, model, epoch)
            sum_mse += recent_mse
            sum_mae += recent_mae
        average_mse = sum_mse / (args.rate_test / args.batch_size)
        average_mae = sum_mae / (args.rate_test / args.batch_size)

        print('average_mse: ', average_mse)
        print('average_mse: ', average_mae)

        is_best = average_mae < best_mae
        best_mae = min(average_mae, best_mae)
        if is_best:
            torch.save(model.state_dict(), model_path)

        print('best_mse: ', best_mae)


if __name__ == '__main__':
    main()
