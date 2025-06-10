import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import time
import copy
from tqdm import tqdm
from model import *
from model import math_layer as ML
from model import math_layerNoS as MLNoS
from preprocess import *
import copy
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import collections

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import r2_score

def setup(rank, world_size, model):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(0)

def cleanup():
    dist.destroy_process_group()

                
def ZUinitialize(model):
    Z = []
    U = []
    for name, para in model.named_parameters():
        if name.split('.')[-1] == 'weight':
            z = para.detach().cpu().clone()
            u = torch.zeros(z.shape)
            Z.append(z)
            U.append(u)
    return Z, U


def admm_loss(rho, model, Z, U, output, target, criterion=F.mse_loss, device = 'cuda:3', l1_pcen = False, Z2 = None, U2 = None):
    loss = criterion(output, target)
    i = 0
    Z = [z.to(device) for z in Z]
    if l1_pcen:
        Z2 = [z.to(device) for z in Z2]
        U2 = [u.to(device) for u in U2]
    U = [u.to(device) for u in U]
    for name, para in model.named_parameters():
        if name.split('.')[-1] == 'weight':
            u = U[i].to(device)
            z = Z[i].to(device)
            if not l1_pcen:
                loss += rho/2*(para - z + u).norm()
            else:
                z2 = Z2[i].to(device)
                u2 = U2[i].to(device)
                loss += rho/2* (0.5*(para - z + u).norm() + 0.5*(para - z2 + u2).norm())
            i += 1
    return loss


def Wupdate(model):
    W = [para.detach().cpu().clone() for name, para in model.named_parameters() if name.split('.')[-1] == 'weight']
    return W


def Zupdate(W, U, thres_admm):
    new_Z = []
    for i in range(len(W)):
        z = W[i] + U[i]
        under_thres = abs(z) < thres_admm
        z.data[under_thres] = 0
        new_Z.append(z)
    return new_Z


def Zupdate_pcen(W, U, pcen_admm):
    new_Z = []
    for i in range(len(W)):
        z = W[i] + U[i]
        pcen = np.percentile(abs(z), 100*pcen_admm)
        under_thres = abs(z) < pcen
        z.data[under_thres] = 0
        new_Z.append(z)
    return new_Z

def update_Z_l1(W, U, rho, alpha):
    new_Z = []
    delta = alpha / rho
    for i in range(len(W)):
        z = W[i] + U[i]
        new_z = z.clone()
        if (z > delta).sum() != 0:
            new_z[z > delta] = z[z > delta] - delta
        if (z < -delta).sum() != 0:
            new_z[z < -delta] = z[z < -delta] + delta
        if (abs(z) <= delta).sum() != 0:
            new_z[abs(z) <= delta] = 0
        new_Z.append(new_z)
    return new_Z


def Uupdate(U, W, Z):
    new_U = [U[i] + W[i] - Z[i] for i in range(len(U))]
    return new_U


def cal_sparsity(model, thres=0.01):
    num = []
    num_zeros = []
    for name, para in model.named_parameters():
        if name.split('.')[-1] == 'weight':
            w = para.detach().cpu().clone()
            under_thres = abs(w) < thres
            num_zeros.append(int(torch.sum(under_thres)))
            num.append(w.shape[0]*w.shape[1])
    return sum(num_zeros)/sum(num)


def cal_sparsity_node(model, thres=0.01):
    num = []
    num_node = 0
    num_zeros = []
    num_zeros_node = []
    for name, para in model.named_parameters():
        if name.split('.')[-1] == 'weight':
            w = para.detach().cpu().clone()
            under_thres = abs(w) < thres
            num_zeros.append(int(torch.sum(under_thres)))
            num.append(w.shape[0]*w.shape[1])
        if name.split('.')[0] == 'bridges':
            ww = para.detach().cpu().clone()
            under_thres_node = abs(ww) < thres
            num_zeros_node.append(int(torch.sum(under_thres_node)))
            num_node += 1
    print(f"node sparsity={sum(num_zeros_node)/num_node}")
    return sum(num_zeros)/sum(num)


def admm_train(dataloader_train, dataloader_test, model, criterion, optimizer, num_epochs, dataset_size, l1_lambda = 0.001, rho = 0.2, admm_thres=0.3, pcen_avai=False, pcen=0.8, test_only=False, decay = 0.9, simple = False, freeze = False, name='', alpha = 5e-4, l1_update = False, test_skipper = False, par = False, rank = 0, world_size = 1, ill_regularization = 1, l1_pcen = False,run_name='run1',device = 'cuda:3'):
        
    if par:
        setup(rank, world_size,model)
        device = torch.device(f'cuda:{rank}')
        model.to(device)
        model = DDP(model, device_ids=[rank])
    else:
        model.to(device)
    regu = torch.tensor(ill_regularization)#0.1)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
    dataloaders = {'train': dataloader_train, 'test': dataloader_test}
    loss_re = {'train': [], 'test': []}
    acc_re = {'train': [], 'test': []}
    r2_re = {'train': [], 'test': []}
    stop = 0
    flag = 0
    if not l1_pcen:
        Z,U = ZUinitialize(model)
    else:
        Z1,U1 = ZUinitialize(model)
        Z2,U2 = ZUinitialize(model)
    test_count = {}
    test_counter = 0
    if not test_only:
        phases = ['train', 'test']
    else:
        phases = ['test']
    gradients = {f'neuron_{i + 1}': [] for i in range(4)}
    #####  reweighting code
    # weight_masks = {}
    # for name1, param in model.named_parameters():
    #     if 'weight' in name1:
    #         weight_masks[name1] = torch.ones_like(param, requires_grad=False)
    #####

    for epoch in range(num_epochs):
        print(cal_sparsity(model))
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()

            else:
                if (test_skipper) and (test_counter < test_skipper):
                    test_counter+=1
                    continue

                test_counter = 0
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            running_ss_tot = 0.0
            running_ss_res = 0.0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    if not l1_pcen:
                        loss = admm_loss(rho, model, Z, U, outputs, labels, criterion, device)
                    else:
                        loss = admm_loss(rho, model, Z1, U1, outputs, labels, criterion, device, True, Z2, U2)
                    ##### reweighted part
                    # l1_loss=0
                    # for name1, param in model.named_parameters():
                    #     if 'weight' in name1:
                    #         weight_mask_current = weight_masks[name1]
                    #         l1_loss += torch.sum(torch.abs(param) * weight_mask_current)
                    # l1_regularization = sum(p.abs().sum() for p in model.parameters())
                    l1_regularization = sum(p.abs().sum() for p in model.parameters())
                    loss += l1_lambda*l1_regularization
                    # loss += l1_lambda * l1_loss
                    if not simple:
                        if not par:
                            for i in range(len(model.model)-1):
                                loss += regu*model.model[i].act.loss
                        else:
                            for i in range(len(model.module.model)-1):
                                loss += regu*model.module.model[i].act.loss

                    if phase == 'train':

                        if freeze:
                            for module in model.model:
                                if hasattr(module.fc, 'weight'):  
                                    module.fc.weight.retain_grad()

                            for module in model.adjuster:
                                if hasattr(module, 'weight'):
                                    module.weight.retain_grad()

                            for module in model.bridges:
                                for k in range(len(model.bridges[0].node)):
                                    if hasattr(module.node[k], 'weight'):      
                                        module.node[k].weight.retain_grad()

                            loss.retain_grad()
                            
                        loss.backward()


                        if freeze:
                            for module in model.model:

                                if hasattr(module.fc, 'weight'):
                                    
                                    mask = module.fc.weight == 0
                                    if not module.fc.weight.requires_grad:
                                        module.fc.weight.retain_grad()
                                    with torch.no_grad():
                                        module.fc.weight.grad[mask] = 0
                                        module.fc.weight.data[mask] = 0 


                            for module in model.adjuster:
                                if hasattr(module, 'weight'):
                                    
                                    mask = module.weight == 0
                                    if not module.weight.requires_grad:
                                        module.weight.retain_grad()
                                    with torch.no_grad():
                                        module.weight.grad[mask] = 0
                                        module.weight.data[mask] = 0


                            for module in model.bridges:
                                for k in range(len(model.bridges[0].node)):
                                    if hasattr(module.node[k], 'weight'):
                                        
                                        mask = module.node[k].weight == 0
                                        if not module.node[k].weight.requires_grad:
                                            module.node[k].weight.retain_grad()
                                        with torch.no_grad():
                                            module.node[k].weight.grad[mask] = 0
                                            module.node[k].weight.data[mask] = 0

                                    
                        optimizer.step()

                        ##### reweighting step (updating weight_masks)
                        # with torch.no_grad():
                        #     for name1, param in model.named_parameters():
                        #         if 'weight' in name1:
                        #             weight_masks[name1] = 1.0 / (torch.abs(param) + 1e-6)
                        #####
                if test_only:
                    x0_L = 5e-3
                    x0_E = 4
                    if not simple:
                        for namet, layer in model.model.named_children():
                            if isinstance(layer, ML) or isinstance(layer, MLNoS):

                                tensor_L = layer.activations['fc'][:,-1]
                                tensor_E = layer.activations['fc'][:,1]
                                
                                mask = (tensor_L < x0_L) & (tensor_L != 0)

                                count_greater_than_threshold = mask.sum(dim=0)
    
                                mask = tensor_E > x0_E
                                count_greater_than_thresholdE = mask.sum(dim=0)
                                
                                if namet+"L" not in test_count:
                                    test_count[namet+"L"] = count_greater_than_threshold
                                    test_count[namet+"E"] = count_greater_than_thresholdE
                                else:
                                    test_count[namet+"L"] =  test_count[namet+"L"] + count_greater_than_threshold
                                    test_count[namet+"E"] =  test_count[namet+"E"] + count_greater_than_thresholdE
                
                    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((abs(outputs - labels.data) / abs(labels.data)) < 0.025)

                ss_tot = torch.sum((labels.data - torch.mean(labels.data)) ** 2)
                ss_res = torch.sum((labels.data - outputs) ** 2)
                running_ss_tot += ss_tot.item()
                running_ss_res += ss_res.item()
                
            W = Wupdate(model)
            if l1_pcen:
                Z1 = Zupdate_pcen(W, U1, pcen)
                Z2 = update_Z_l1(W, U2, rho, alpha)
                U1 = Uupdate(U1, W, Z1)
                U2 = Uupdate(U2, W, Z2)
            elif pcen_avai:
                Z = Zupdate_pcen(W, U, pcen)
                U = Uupdate(U, W, Z)
            elif l1_update:
                Z = update_Z_l1(W, U, rho, alpha)
                U = Uupdate(U, W, Z)
            else:
                Z = Zupdate(W, U, admm_thres)
                U = Uupdate(U, W, Z)
            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects / dataset_size[phase]
            epoch_r2 = 1 - (running_ss_res / running_ss_tot)
            
            loss_re[phase].append(epoch_loss)
            acc_re[phase].append(float(epoch_acc.cpu().detach().numpy()))
            r2_re[phase].append(epoch_r2)
            print('{} Loss: {:.4f} Acc: {:.4f} R2: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_r2))
            if epoch_loss != epoch_loss:
                print("OK", running_loss, "restart training!!")
                stop = 1
                return stop, flag, loss_re, acc_re, r2_re
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        if epoch == num_epochs - 1:
            flag = 1

        scheduler.step()

    best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(best_model_wts, run_name+name + 'best_model_l1.pth')
    if test_only:
        return stop, flag, loss_re, acc_re,r2_re, test_count
    return stop, flag, loss_re, acc_re, r2_re



def admm_test(dataloader_train, dataloader_test, min_val,max_val,model, criterion, optimizer, num_epochs, dataset_size,
               l1_lambda=0.001, rho=0.2, admm_thres=0.3, pcen_avai=False, pcen=0.8, test_only=False, decay=0.9,
               simple=False, freeze=False, name='', alpha=5e-4, l1_update=False, device='cpu', mode='edge',n_input=3, ill_regularization = 1,logT=False,run_name='run1'):
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    regu = torch.tensor(ill_regularization)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
    dataloaders = {'train': dataloader_train, 'test': dataloader_test}
    loss_re = {'train': [], 'test': []}
    acc_re = {'train': [], 'test': []}
    score_re = {'train': [], 'test': []}
    stop = 0
    flag = 0
    Z, U = ZUinitialize(model)
    test_count = {}

    if not test_only:
        phases = ['train']
    else:
        phases = ['test']
    for epoch in range(num_epochs):
        inputs_ls=[]
        labels_ls = []
        outputs_ls = []
        print(cal_sparsity(model))
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()

            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            score = 0.0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                inputs_ls.append(inputs)
                labels = labels.to(device)
                labels_ls.append(labels)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    outputs_ls.append(outputs)
                    loss = admm_loss(rho, model, Z, U, outputs, labels, criterion, device=device)
                    l1_regularization = sum(p.abs().sum() for p in model.parameters())
                    loss += l1_lambda * l1_regularization
                    if not simple:
                        for i in range(len(model.model) - 1):
                            loss += regu * model.model[i].act.loss

                    if phase == 'train':

                        loss.backward()

                        if freeze:
                            for module in model.model:

                                if hasattr(module.fc, 'weight'):
                                    mask = module.fc.weight == 0
                                    module.fc.weight[mask].grad = torch.zeros_like(
                                        module.fc.weight[mask])
                                    module.fc.weight.data[mask] = 0

                            for module in model.adjuster:
                                if hasattr(module, 'weight'):
                                    mask = module.weight == 0
                                    module.weight[mask].grad = torch.zeros_like(
                                        module.weight[mask])
                                    module.weight.data[mask] = 0

                            for module in model.bridges:
                                for k in range(len(model.bridges[0].node)):
                                    if hasattr(module.node[k], 'weight'):
                                        mask = module.node[k].weight == 0
                                        module.node[k].weight[mask].grad = torch.zeros_like(
                                            module.node[k].weight[mask])
                                        module.node[k].weight.data[mask] = 0

                        optimizer.step()
                if test_only:

                    x0_L = 5e-3
                    x0_E = 4
                    if not simple:
                        for name, layer in model.model.named_children():

                            if isinstance(layer, ML) or isinstance(layer, MLNoS):

                                tensor_L = layer.activations['fc'][:, -1]
                                tensor_E = layer.activations['fc'][:, 1]
                                mask = tensor_L < x0_L  # In illegal range

                                count_greater_than_threshold = mask.sum(dim=0)

                                mask = tensor_E > x0_E  # In illegal range
                                count_greater_than_thresholdE = mask.sum(dim=0)

                                if name + "L" not in test_count:
                                    test_count[name + "L"] = count_greater_than_threshold
                                    test_count[name + "E"] = count_greater_than_thresholdE
                                else:
                                    test_count[name + "L"] = test_count[name + "L"] + count_greater_than_threshold
                                    test_count[name + "E"] = test_count[name + "E"] + count_greater_than_thresholdE

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(abs((outputs - labels.data) / labels.data) < 0.025)
            bucket_list = []
            inputs_ls=torch.concatenate(inputs_ls, axis=0)
            labels_ls = torch.concatenate(labels_ls, axis=0)
            outputs_ls = torch.concatenate(outputs_ls, axis=0)
            for i in range(n_input):
                bin_edges=torch.linspace(min_val[i], max_val[i], 9)
                bucket_arr=torch.bucketize(inputs_ls[:, i] * max_val[i], bin_edges)-1
                bucket_arr[bucket_arr==-1]=0
                bucket_list.append(bucket_arr)

            indices = torch.column_stack(bucket_list)
            error = np.full((8,) * n_input, np.nan)
            from itertools import product
            value_range = range(0, 8)
            combinations = product(value_range, repeat=n_input)
            for combo in combinations:
                mask = (indices == torch.tensor(combo)).all(dim=1)
                if not mask.any():

                    continue
                loc = np.where(mask)[0]

                labels_e=labels_ls[loc]
                outputs_e = outputs_ls[loc]
                score = r2_score(labels_e.numpy(), outputs_e.numpy())
                error[combo]=score

            score_total = r2_score(np.array(labels_ls), np.array(outputs_ls))
            W = Wupdate(model)
            if pcen_avai:
                Z = Zupdate_pcen(W, U, pcen)
            elif l1_update:
                Z = update_Z_l1(W, U, rho, alpha)
            else:
                Z = Zupdate(W, U, admm_thres)
            U = Uupdate(U, W, Z)
            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects / dataset_size[phase]
            epoch_score = score / dataset_size[phase]
            loss_re[phase].append(epoch_loss)
            acc_re[phase].append(float(epoch_acc.cpu().detach().numpy()))
            score_re[phase].append(epoch_score)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, score))

            if epoch_loss != epoch_loss:
                print("OK", running_loss, "restart training!!")
                stop = 1
                return stop, flag, loss_re, acc_re
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        if epoch == num_epochs - 1:
            flag = 1
        torch.save(best_model_wts, run_name+name + 'best_model_l1.pth')
        scheduler.step()
    if test_only:
        return stop, flag, loss_re, acc_re, test_count, error, score_total
    return stop, flag, loss_re, acc_re
