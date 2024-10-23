import random

import nibabel as nib
import os
import torch
import numpy as np
import pandas as pd

import args_parser

args = args_parser.args_parse()

def build_datasets():

    directory = './Data/training'
    directory_label = './Data/label'
    p = args.num_subspace

    start_idx = args.data_start
    end_idx = args.data_end


    def file_exists(file_path):
        return os.path.isfile(file_path)


    # def read_nifti_as_tensor(file_path):
    #     image = nib.load(file_path).get_fdata()
    #     return torch.tensor(image, dtype=torch.float32)
    label_file_name = f'Label_new.csv'
    label_path = os.path.join(directory_label, label_file_name)
    df = pd.read_csv(label_path)

    data_list = []
    label_list = []
    # i=0
    for idx in range(start_idx, end_idx + 1):
        file_name = f'image{idx:03}.nii.gz'
        file_path = os.path.join(directory, file_name)
        temp = f'image{idx:03}'
        filtered_df = df[df['image_name'].apply(lambda x: x == temp if isinstance(x, str) else False)]
        label_list.extend(filtered_df.to_dict('records'))
        # print(label_list[i]['real_age'])
        # # i += 1


        if file_exists(file_path):

            image = nib.load(file_path).get_fdata()
            image = image[:, 150:-169:, :]
            H, W, S, = image.shape
            image = image.transpose(2,0,1).reshape(image.shape[2],-1)
            U0, S, V = np.linalg.svd(np.dot(image,image.T))
            U0 = U0[:, 0:int(p)]
            sub_represent = np.tensordot(U0.T, image, axes=([1], [0])).transpose(1,0).reshape(H,W,p)
            data_list.append(sub_represent)
        else:
            print(f'File {file_name} does not exist in {directory}.')


        #creat test data
    list_size = len(label_list)
    val_indices = random.sample(range(list_size),args.rate_test)
    print(val_indices)
    test_list = []
    test_ref = []
    test_gender = []
    test_name_list = []
    for i in val_indices:
        test_list.append(data_list[i])
        test_ref.append(label_list[i]['real_age'])
        test_gender.append(label_list[i]['gender'])
        test_name_list.append(label_list[i]['image_name'])

    with open('test_data.txt', 'a') as f:
        f.write(str(test_name_list) + '\n')

    train_list = [data_list[i] for i in range(list_size) if i not in val_indices]
    train_ref = [label_list[i]['real_age'] for i in range(list_size) if i not in val_indices]
    train_gender = [label_list[i]['gender'] for i in range(list_size) if i not in val_indices]

    train_gender_numeric = np.array([1 if g == 'M' else 0 for g in train_gender])
    test_gender_numeric = np.array([1 if g == 'M' else 0 for g in test_gender])

    train_list = torch.from_numpy(np.stack(train_list, axis=0)).unsqueeze(0).permute(1, 0, 4, 2, 3)
    train_ref = torch.from_numpy(np.stack(train_ref, axis=0))
    train_gender = torch.from_numpy(np.stack(train_gender_numeric, axis=0))


    test_list = torch.from_numpy(np.stack(test_list, axis=0)).unsqueeze(0).permute(1, 0, 4, 2, 3)
    test_ref = torch.from_numpy(np.stack(test_ref, axis=0))
    test_gender = torch.from_numpy(np.stack(test_gender_numeric, axis=0))

    return [train_list,train_ref], [test_list,test_ref], train_gender, test_gender




