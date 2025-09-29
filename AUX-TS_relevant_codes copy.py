from functools import reduce
import math
import os
import time
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F

from config import Config

# Delete
class Classifier(nn.Module):
    def __init__(self, n_hid):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.linear = nn.Linear(n_hid, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

    def __repr__(self):
        return '{}(n_hid={}, 1)'.format(
            self.__class__.__name__, self.n_hid)
    

class Weight(nn.Module):
    def __init__(self, in_channel, hidden, out_channel, act_type='sigmoid'):
        super(Weight, self).__init__()
        # self.linear1 = MetaLinear(in_channel, hidden)
        # self.linear2 = MetaLinear(hidden, out_channel)

        self.linear1 = nn.Linear(in_channel, hidden)
        self.linear2 = nn.Linear(hidden, out_channel)
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leaky':
            self.act = nn.LeakyReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'softplus':
            self.act = nn.Softplus()
        else:
            raise ValueError('unknown activation type!' + act_type)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.act(x)
        return x


'''
    !!!!!!!!!! Azért van mindig egy .._meta, mert a tanítás során az eredeti modellt lemásolja a 
    .._meta-ba amivel tud próbálkozni a k_fold során !!!!!!!!!!!!!!!!!


    Initialize model for the primary task
    The primary task is for node classification => to wich class does it belong
'''
# classifier = Classifier(args.n_hid, graph.y.max().item() + 1)
# classifier_meta = Classifier(args.n_hid, graph.y.max().item() + 1)


'''
    Initialize model for the auxiliary tasks
    Here the model (gnn) can be anything since it's only used for node embeding => i will use my simple gnn
    
    MT_GNN is a complex gnn for multi task gnn learning
'''

''''
    attr_decoder: for node feat distance 
    i have to do the node classification task like this !!!!!!!!!!!!!!!!!

    def feat_loss(self, reps, out):
        return -self.attr_decoder(reps, out)

'''

mt_gnn = _
mt_gnn_meta = _

# A kezdeti embedding vektorokat álítja elő, hogy a node típusok featureinek átlagát veszi => ez a több típusú node-ok miatt van
node_features = any
init_emb = node_features.mean(dim=0).detach()

'''
Initialize weight model for multi-tasks
'''
vnet = Weight(5, Config.v_emb_dim, 1, Config.v_act_type)
optimizer_v = torch.optim.Adam(vnet.parameters(), lr=Config.v_lr, weight_decay=1e-3)

'''
Optimizer
''' 
params = mt_gnn.parameters()
optimizer = torch.optim.AdamW(params, weight_decay = 1e-2, eps=1e-06, lr = Config.mt_lr)

params_meta = mt_gnn_meta.parameters()
optimizer_meta = torch.optim.AdamW(params_meta, weight_decay = 1e-2, eps=1e-06, lr = Config.mt_lr)

# Az optimizer tanulási rátájánnak csökkentéséhez
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= Config.epochs, eta_min=1e-6)
scheduler_meta = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_meta, T_max=Config.epochs, eta_min=1e-6)

train_data_pr =  _ 
valid_data_pr = _
train_data_au =  _ 

# prepare training
best_val = 0
train_step = 0
train_step_meta = 0
res = []

# a shared paramsban benne van a classifier is ami a pr_task-é => ez most is így van
private_param_name = [p[0] for p in mt_gnn.named_parameters() if 'aux_classifiers' in p[0] and p[1].requires_grad]
share_param_name = []
for n, p in mt_gnn.named_parameters():
    if n not in private_param_name and p.requires_grad:
        share_param_name.append(n)

cos_ = nn.CosineSimilarity()
dummy_flag = False # aktiváld a gradienst
dummy_flag_meta = False

st = time.time()

# pr osztályozáshoz: log-softmax után használható
# criterion = nn.NLLLoss(reduce=False)

# get the cosine similarity for task gradients
def get_cos(params, mt_gnn, optimizer, share_param_name, cos_, loss_pr_mean, loss_attr_mean):
    optimizer.zero_grad()
    loss_pr_mean.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(params, Config.clip)
    pr_share_grad = record_grad([mt_gnn], share_param_name)
    pr_grad_flat = torch.cat([p.clone().flatten() for _, p in pr_share_grad.items()])

    # gradients for node prediction
    optimizer.zero_grad()
    loss_attr_mean.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(params, Config.clip)
    aux_share_grad = record_grad([mt_gnn], share_param_name)
    aux_grad_flat = torch.cat([p.clone().flatten() for n, p in aux_share_grad.items()])

    pr_aux_cos = cos_(pr_grad_flat.view(1, -1), aux_grad_flat.view(1, -1))

    return pr_aux_cos

def clean_param_name(model_list, name):
    '''
        clean param.grad is None in name(unused param in model)
    '''
    for model in model_list:
        for n, v in zip(model.state_dict(), model.parameters()):
            if v.grad is None and n in name:
                name.remove(n)
                print('remove', n)

def record_grad(model_list, name):
    '''
        record the grad of param with name[list] in model_list

        return: dict
    '''
    ret = {}
    for model in model_list:
        for n, v in zip(model.state_dict(), model.parameters()):
            if n in name and v.grad is not None:
                ret[n] = v.grad.clone().detach()
    return ret
 
for epoch in np.arange(1) + 1:
    mt_gnn.train()
    mt_gnn_meta.train()
    train_pr_losses = []
    train_aux_losses = []

    node_feature, edge_index, x_ids, ylabel = train_data_pr
    data, attr = train_data_au
    node_feature_au, edge_index_au = data
    node_feature_au = node_feature_au.f()
    # mindegyik feature a kezdeti embeding feature lesz
    node_feature_au = init_emb
    
    loss_pr_meta = 0 
    pr_aux_cos = 0
    for j in range(Config.n_fold):
        # meta model, az eredeti modellek másolata
        mt_gnn_meta.load_state_dict(mt_gnn.state_dict())

        # run tasks task
        # loss per example => reduce = false
        loss_pr, loss_aux = mt_gnn_meta.forward(node_feature, edge_index) # a segéd taskok adatait is itt adjuk be 

        # for log
        loss_pr_mean = loss_pr.mean()
        loss_aux_mean = loss_aux.mean()

        #egyszer megy csak bele
        if not dummy_flag_meta:
            optimizer_meta.zero_grad()
            loss = loss_pr_mean + loss_aux_mean
            loss.backward(retain_graph=True)
            clean_param_name([mt_gnn_meta], share_param_name)
            clean_param_name([mt_gnn_meta] , private_param_name)
            dummy_flag_meta = True
        else:
            loss = 0

        pr_aux_cos = get_cos(params_meta, mt_gnn_meta, optimizer_meta, share_param_name, cos_, loss_pr_mean, loss_aux_mean)

        # embeddings for v-net
        loss_pr_emb = torch.stack((loss_pr, \
                torch.ones([len(loss_pr)], device=loss_pr.device), \
                torch.zeros([len(loss_pr)], device=loss_pr.device), \
                torch.zeros([len(loss_pr)], device=loss_pr.device), \
                torch.full([len(loss_pr)], 1.0, device=loss_pr.device),
            ))
        loss_aux_emb = torch.stack((loss_aux, \
                torch.zeros([len(loss_aux)], device=loss_pr.device), \
                torch.zeros([len(loss_aux)], device=loss_pr.device), \
                torch.ones([len(loss_aux)], device=loss_pr.device), \
                torch.full([len(loss_aux)], pr_aux_cos.item(), device=loss_pr.device)
            ))

        loss_pr_emb = loss_pr_emb.transpose(1,0)
        loss_aux_emb = loss_aux_emb.transpose(1,0)

        # compute weight
        v_pr = vnet(loss_pr_emb)
        v_aux = vnet(loss_aux_emb)

        # compute loss
        loss_pr_avg = (loss_pr * v_pr).mean()
        loss_aux_avg = (loss_aux * v_aux).mean()
        loss_meta = loss_pr_avg  + loss_aux_avg

        # one step update of model parameter (fake) (Eq.6)
        optimizer_meta.zero_grad()
        loss_meta.backward()
        torch.nn.utils.clip_grad_norm_(params_meta, Config.clip)
        optimizer_meta.step()
        train_step_meta += 1
        scheduler_meta.step(train_step_meta)

        # primary loss with updated parameter (Eq.7)
        loss_pr_meta += mt_gnn_meta.forward(node_feature, edge_index)

    # in each batch

    # backward and update v-net params (Eq.9)
    optimizer_v.zero_grad()
    loss_pr_meta.backward()
    optimizer_v.step()

    # with the updated weight, update model parameters (true) (Eq.8)
    node_feature, edge_index, x_ids, ylabel = train_data_pr
    # data, attr, (start_idx, end_idx) = train_data_au
    # node_feature_au, edge_index_au = data
    # node_feature_au = node_feature_au.detach()
    # node_feature_au[start_idx: end_idx] = init_emb

    loss_pr, loss_aux = mt_gnn.forward(node_feature, edge_index) # add aux task data too

    # for log
    loss_pr_mean = loss_pr.mean()
    loss_aux_mean = loss_aux.mean()
    if not dummy_flag:
        optimizer.zero_grad()
        loss = loss_pr_mean + loss_aux_mean
        loss.backward(retain_graph=True)
        clean_param_name([mt_gnn], share_param_name)
        clean_param_name([mt_gnn], private_param_name)
        dummy_flag = True

    pr_aux_cos = get_cos(params, mt_gnn, optimizer, share_param_name, cos_, loss_pr_mean, loss_aux_mean)

    print('aux cos: %f'%(pr_aux_cos.item()))

    # embeddings for v-net
    # shape = (5, batch size)   
    loss_pr_emb = torch.stack((loss_pr, \
                torch.ones([len(loss_pr)], device=loss_pr.device), \
                torch.zeros([len(loss_pr)], device=loss_pr.device), \
                torch.zeros([len(loss_pr)], device=loss_pr.device), \
                torch.full([len(loss_pr)], 1.0, device=loss_pr.device),
            ))
    loss_aux_emb = torch.stack((loss_aux, \
            torch.zeros([len(loss_aux)], device=loss_pr.device), \
            torch.zeros([len(loss_aux)], device=loss_pr.device), \
            torch.ones([len(loss_aux)], device=loss_pr.device), \
            torch.full([len(loss_aux)], pr_aux_cos.item(), device=loss_pr.device)
        ))
    # embeddings for v-net
    #shape = (batch size, 5)
    loss_pr_emb = loss_pr_emb.transpose(1, 0)
    loss_aux_emb = loss_aux_emb.transpose(1, 0)

    # compute weight
    with torch.no_grad():
        v_pr = vnet(loss_pr_emb)
        v_aux = vnet(loss_aux_emb)

    # compute loss
    loss_pr_avg = (loss_pr * v_pr).mean()
    loss_aux_avg = (loss_aux * v_aux).mean()
    loss_pr_avg_weighted = loss_pr.mean()
    loss_aux_avg_weighted = loss_aux.mean()

    print((
                "Epoch: %d   Train Loss Pr: %.2f  Train Loss Aux: %.2f  Pr_Weight_Mean: %.4f Aux_Weight_Mean: %.4f Pr_Weight_Std: %.4f Aux_Weight_Std: %.4f ") %
            (epoch, loss_pr_avg_weighted, loss_aux_avg_weighted, v_pr.mean().item(), v_aux.mean().item(), v_pr.std().item(), v_aux.std().item()))

    # total loss
    loss = loss_pr_avg + loss_aux_avg
    train_pr_losses += [loss_pr_avg_weighted.cpu().detach().tolist()]
    train_aux_losses += [loss_aux_avg_weighted.cpu().detach().tolist()]

    # optimize model parameters
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, Config.clip)
    optimizer.step()
    train_step += 1
    scheduler.step(train_step)
    del res, loss_pr, loss_aux, loss_pr_emb, loss_aux_emb, v_pr, v_aux
    del loss_pr_mean, loss_aux_mean, loss_pr_avg_weighted, loss_aux_avg_weighted


# OK ------------------------------------------------------
    '''
        Valid with only primary task (2017 <= time <= 2017)

        in each epoch
    '''
    with torch.no_grad():
        valid_loss_pr_mean = 0
        valid_loss_link_mean = 0
        valid_loss_attr_mean = 0
        valid_res = []
        loss = 0
        node_feature, edge_index, x_ids, ylabel = valid_data_pr
        loss += mt_gnn.forward(node_feature, edge_index, )
        # res  = classifier.forward(node_rep[x_ids])
        # loss += criterion(res, ylabel.to(device)).mean() / args.valid_batch #scalar
        # loss = criterion(res, ylabel.to(device))
    
        '''
            Calculate Valid F1. Update the best model based on highest F1 score.
        '''
        valid_res += [f1_score(res.argmax(dim=1).cpu().tolist(), ylabel.tolist(), average='micro')]
        
        valid_f1 = np.average(valid_res)
        if valid_f1 > best_val:
            best_val = valid_f1
            # save the best model => here was model_pr
            torch.save( _ ,
                    os.path.join(args.model_dir, args.task_name + '_' + args.conv_name + '_' + args.model_add_name))
            print('UPDATE!!!')
        
        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train pr Loss: %.2f Train aux Loss: %.2f  Valid Loss: %.2f  Valid F1: %.4f") % \
              (epoch, optimizer.param_groups[0]['lr'], np.average(train_pr_losses), np.average(train_aux_losses),\
                    loss.cpu().detach().tolist(), valid_f1))
        del res, loss
    del train_data_pr, valid_data_pr, train_data_au


    '''
        Test 
    '''
    if epoch > 0 and epoch % 5 == 0:
        best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name + '_' + args.model_add_name), map_location=CUDA_STR)
        best_model.eval()
        gnn_best, classifier_best = best_model
        with torch.no_grad():
            test_res = []
            for _ in range(10):
                node_feature, edge_index, x_ids, ylabel = ... # teszt gráf
                paper_rep = gnn_best.forward(node_feature, edge_index)[x_ids]
                res = classifier_best.forward(paper_rep)
                test_f1 = f1_score(res.argmax(dim=1).cpu().tolist(), ylabel.tolist(), average='micro')
                test_res += [test_f1]
        print('Best Test F1: %.4f' % np.average(test_res))

