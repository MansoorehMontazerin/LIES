import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random
import math
import torch.nn as nn

class ExpLoss(nn.Module):
    def __init__(self):
        super(ExpLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.sum(torch.exp(torch.abs(output - target))) / output.shape[0]
        return loss

class SineDataset(Dataset):
    def __init__(self, start=1, end=5, num_samples=1000000):
        self.x = np.linspace(start, end, num_samples)
        self.y = 50*np.sin(self.x)
        self.ones = np.ones(num_samples)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
         return np.array([self.x[idx], self.ones[idx]]).astype('float32'), np.array(self.y[idx]).astype('float32')


class custom_data(Dataset):
    def __init__(self, eqn = "I.14.4", logT = True):
        #eqn = "I.14.4"
        self.count = 0
        df = pd.read_csv('./BenchData/Feynman_with_units/' + eqn, header = None, sep = ' ')
        df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
        self.data = {}
        last_col = df.columns[-1]
        if logT:
            for col in df.columns:
                self.data[col] = [math.log(float(i)+1e-10) for i in df[col]]
                self.count += 1
        else:
            for col in df.columns:
                self.data[col] = [(float(i)+1e-10) for i in df[col]]
                self.count += 1
        self.data[self.count] = [float(1)] * len(self.data[self.count-1])
        self.count += 1

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        inputs = np.array([self.data[col][idx] for col in self.data if col != (self.count-2)]).astype('float32')
        outputs = np.array([self.data[self.count-2][idx]]).astype('float32')
        return inputs, outputs

class custom_data_bias(Dataset):
    def __init__(self, eqn = "I.14.4", logT = True, trim = 0.1, bias = None):
        upper,lower,X=bias
        if logT:
            upper=np.exp(upper)
            lower=np.exp(lower)
        self.count = 0
        df = pd.read_csv('./BenchData/Feynman_with_units/' + eqn, header=None, sep=' ')
        df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
        n_biased = int(X * len(df) * trim)
        n_unbiased = int(len(df) * trim) - n_biased
        conditions = True
        for i in range(df.shape[1]-1):
            conditions &= (df.iloc[:, i] >= float(lower[i])) & (df.iloc[:, i] <= float(upper[i]))
    
        filtered_df = df[conditions]
        unfiltered_df = df.drop(filtered_df.index)
        sampled_df_biased = filtered_df.sample(n=n_biased, random_state=42,replace=True)
        sampled_df_unbiased = unfiltered_df.sample(n=int(len(df) * trim), random_state=42)
        dataset = pd.concat([sampled_df_biased , sampled_df_unbiased], axis=0, ignore_index=True)
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        self.data = {}

        if logT:
            for col in df.columns:
                self.data[col] = [math.log(float(i) + 1e-10) for i in dataset[col]]
                self.count += 1
        else:
            for col in df.columns:
                self.data[col] = [(float(i) + 1e-10) for i in dataset[col]]
                self.count += 1
        self.data[self.count] = [float(1)] * len(self.data[self.count - 1])
        self.count += 1
    
    def __len__(self):
        return len(next(iter(self.data.values())))
    
    def __getitem__(self, idx):
        inputs = np.array([self.data[col][idx] for col in self.data if col != (self.count - 2)]).astype('float32')
        outputs = np.array([self.data[self.count - 2][idx]]).astype('float32')
        return inputs, outputs

        



def min_max_normalize(tensor, min_vals, max_vals):
    return tensor / (max_vals)

def compute_min_max(dataset):
    data_stack = torch.stack([torch.tensor(data) for data, _ in dataset], dim=0)
    output_stack = torch.stack([torch.tensor(data) for _, data in dataset], dim=0)
    min_vals = data_stack.min(dim=0).values
    max_vals = data_stack.max(dim=0).values

    min_vals_output = data_stack.min(dim=0).values
    max_vals_output = data_stack.max(dim=0).values
    
    return min_vals, max_vals, min_vals_output, max_vals_output

def normalize_dataset(dataset, min_vals, max_vals):
    for i in range(len(dataset)):
        data, label = dataset[i]
        dataset[i] = (min_max_normalize(torch.tensor(data), min_vals, max_vals).numpy(), label)  # Normalize data only
    return dataset
    
def normalize_dataset_output(dataset, min_vals, max_vals):
    for i in range(len(dataset)):
        data, label = dataset[i]
        dataset[i] = (data, min_max_normalize(torch.tensor(label), min_vals, max_vals).numpy())  # Normalize data only
    return dataset


def dataloader_custom(eqn = "I.14.4", batch_size=64, n_workers=0, rate=0.8, norm = False, logT = True, trim = False, bias = False, biased_features=None):
    dataset = custom_data(eqn, logT)
    idx = [i for i in range(len(dataset))]
    random.shuffle(idx)
    dataset_train = [dataset[i] for i in idx[:int(rate * len(dataset))]]
    dataset_test = [dataset[i] for i in idx[int(rate * len(dataset)):]]
    min_val, max_val, min_vals_output, max_vals_output = None, None, None, None

    if bias:
        (lower_bound, upper_bound), X = bias

        biased_samples = []
        unbiased_samples = []
        i = 0
        for sample in dataset_train:
            features = sample[0][:-1]

            if biased_features:
                selected_features = features[biased_features]
            else:
                selected_features = features[:-1]
                
            if (features >= lower_bound).all() and (features <= upper_bound).all():
                biased_samples.append(sample)
            else:
                unbiased_samples.append(sample)

        if trim:
            n_biased = int(X * len(dataset_train)*trim)
            n_unbiased = int(len(dataset_train)*trim) - n_biased
            selected_biased = random.sample(biased_samples, min(n_biased, len(biased_samples)))

            if len(selected_biased) < n_biased:
                remaining_vals = n_biased - len(selected_biased) 
                print(len(biased_samples),remaining_vals)
                selected_biased2 = random.sample(biased_samples, remaining_vals)
                selected_biased = selected_biased + selected_biased2

            selected_unbiased = random.sample(unbiased_samples, min(n_unbiased, len(unbiased_samples)))
        else:
            n_biased = int(X * len(dataset_train))
            n_unbiased = len(dataset_train) - n_biased
    

            selected_biased = random.sample(biased_samples, min(n_biased, len(biased_samples)))

            if len(selected_biased) < n_biased:
                remaining_vals = n_biased - len(selected_biased) 
                selected_biased2 = random.sample(biased_samples, remaining_vals)
                selected_biased = selected_biased + selected_biased2
            
            selected_unbiased = random.sample(unbiased_samples, min(n_unbiased, len(unbiased_samples)))

        dataset_train = selected_biased + selected_unbiased
        random.shuffle(dataset_train)

    elif trim:
        dataset_train = dataset_train[:int(trim * len(dataset_train))]
        dataset_test = dataset_test[:int(trim * len(dataset_test))]
    if norm:
        if rate !=1:
            min_val, max_val, min_vals_output, max_vals_output = compute_min_max(dataset_test)
            dataset_test = normalize_dataset([data for data in dataset_test], min_val, max_val)

        min_val, max_val, min_vals_output, max_vals_output = compute_min_max(dataset_train)

        dataset_train = normalize_dataset([data for data in dataset_train], min_val, max_val)

    else:
        min_val, max_val, min_vals_output, max_vals_output = compute_min_max(dataset_train)
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=n_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=n_workers)
    if norm:
        return dataloader_train, dataloader_test, min_val, max_val, min_vals_output, max_vals_output
    else:
        return dataloader_train, dataloader_test



def dataloader_custom_bias(eqn="I.14.4", batch_size=64, n_workers=0, rate=0.8, norm=False, logT=True, trim=False, bias=False,
                      biased_features=None):


    dataset = custom_data_bias(eqn, logT, trim, bias=bias)
    idx = [i for i in range(len(dataset))]
    random.shuffle(idx)
    dataset_train = [dataset[i] for i in idx[:int(rate * len(dataset))]]
    dataset_test = [dataset[i] for i in idx[:int(rate * len(dataset))]]
    min_val, max_val, min_vals_output, max_vals_output = None, None, None, None
    if norm:
        min_val, max_val, min_vals_output, max_vals_output = compute_min_max(
            dataset_train)
        dataset_train = normalize_dataset([data for data in dataset_train], min_val, max_val)
        dataset_test = normalize_dataset([data for data in dataset_test], min_val, max_val)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=n_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset), num_workers=n_workers)
    if norm:
        return dataloader_train, dataloader_test, min_val, max_val, min_vals_output, max_vals_output
    else:
        return dataloader_train, dataloader_test



