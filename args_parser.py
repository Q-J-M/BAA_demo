import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-arch', type=str, default='Unet3D',choices=['SimpleNet','Unet3D'])
    parser.add_argument('-gender_num', type=int, default='32')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='end epoch for training')
    parser.add_argument('--model_path', type=str,
                        default='./checkpoints/arch.pkl')
    parser.add_argument('--num_subspace', type=int,
                        default='16',help='the numbers of subspace')
    parser.add_argument('--data_start', type=int, default='20')
    parser.add_argument('--data_end', type=int, default='108')
    parser.add_argument('--rate_test', type=int, default='12')#10
    parser.add_argument('--batch_size', type=int, default='4')#4


    args = parser.parse_args()
    return args