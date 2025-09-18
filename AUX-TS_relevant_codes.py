from functools import reduce
import math
import os
import time
import torch.nn as nn
from warnings import filterwarnings
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F


from GPT_GNN.model_meta import GPT_GNN

args = {}
gnn = any
graph = any
device = any
rem_edge_list = any
types = any
node_feature = any
node_type = any
node_dict = any
target_type = any
gnn_meta = any
repeat_num = any


class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)

    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)
    
class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''

    def __init__(self, n_hid, n_out, temperature=0.1):
        super(Matcher, self).__init__()
        self.n_hid = n_hid
        self.linear = nn.Linear(n_hid, n_out)
        self.sqrt_hd = math.sqrt(n_out)
        self.drop = nn.Dropout(0.2)
        self.cosine = nn.CosineSimilarity(dim=1)
        self.cache = None
        self.temperature = temperature

    def forward(self, x, ty, use_norm=True):
        tx = self.drop(self.linear(x))
        if use_norm:
            return self.cosine(tx, ty) / self.temperature
        else:
            return (tx * ty).sum(dim=-1) / self.sqrt_hd

    def __repr__(self):
        return '{}(n_hid={})'.format(
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
    .._met-ába amivel tud próbálkozni a k_fold során !!!!!!!!!!!!!!!!!


    Initialize model for the primary task
    The primary task is for node classification => to wich class does it belong
'''
classifier = Classifier(args.n_hid, graph.y.max().item() + 1)
classifier_meta = Classifier(args.n_hid, graph.y.max().item() + 1)


model_pr = nn.Sequential(gnn, classifier).to(device)

'''
    Initialize model for the auxiliary tasks
    Here the model (gnn) can be anything since it's only used for node embeding => i will use my simple gnn
    First i should yous GPT_GNN, but if i know how i will rid of it
'''
# A csomópontok közti távolsághoz
attr_decoder = Matcher(gnn.n_hid, gnn.in_dim)

gpt_gnn = GPT_GNN(gnn=gnn, rem_edge_list=rem_edge_list, attr_decoder=attr_decoder, \
                  neg_queue_size=0, types=types, neg_samp_num=args.neg_samp_num, device=device)
gpt_gnn_meta = GPT_GNN(gnn=gnn_meta, rem_edge_list=rem_edge_list, attr_decoder=attr_decoder, \
                  neg_queue_size=0, types=types, neg_samp_num=args.neg_samp_num, device=device)

# A kezdeti embedding vektorokat álítja elő, hogy a node típusok featureinek átlagát veszi => ez a több típusú node-ok miatt van
init_emb = node_feature[node_type == node_dict[target_type][1]].mean(dim=0).detach()

gpt_gnn = gpt_gnn.to(device)

'''
Initialize weight model for multi-tasks
'''
vnet = Weight(5, args.weight_emb_dim, 1, args.acttivation_type).to(device)
optimizer_v = torch.optim.Adam(vnet.parameters(), lr=args.wlr, weight_decay=1e-3)

# Optimizer
# params = list(gpt_gnn.parameters()) + list(model_pr.parameters())
params = list(gpt_gnn.parameters()) + list(classifier.parameters())
optimizer = torch.optim.AdamW(params, weight_decay = 1e-2, eps=1e-06, lr = args.max_lr)

params_meta = list(gpt_gnn_meta.parameters()) + list(classifier_meta.parameters())
optimizer_meta = torch.optim.AdamW(params_meta, weight_decay = 1e-2, eps=1e-06, lr = args.max_lr)

# Az optimizer tanulási rátájánnak csökkentéséhez
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, repeat_num * args.n_batch, eta_min=1e-6)
scheduler_meta = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_meta, repeat_num * args.n_batch, eta_min=1e-6)

# prepare training
best_val = 0
train_step = 0
train_step_meta = 0
res = []

# prepare for cosine similarty =>get the params into two group
link_private_param_name = [p[0] for p in gpt_gnn.named_parameters() if 'params' in p[0] and p[1].requires_grad]
attr_private_param_name = [p[0] for p in gpt_gnn.named_parameters() if 'attr_decoder' in p[0] and p[1].requires_grad]
private_param_name = link_private_param_name + attr_private_param_name
share_param_name = []
for m in [gpt_gnn, classifier]:
    for n, p in m.named_parameters():
        if n not in private_param_name and p.requires_grad:
            share_param_name.append(n)

cos_ = nn.CosineSimilarity()
dummy_flag = False # aktiváld a gradienst
dummy_flag_meta = False

st = time.time()
criterion = nn.NLLLoss(reduce=False)

# get the cosine similarity for task gradients
def get_cos(args, device, classifier, params, gpt_gnn, optimizer, share_param_name, cos_, loss_pr_mean, loss_link_mean, loss_attr_mean):
    optimizer.zero_grad()
    loss_pr_mean.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(params, args.clip)
    pr_share_grad = record_grad([gpt_gnn, classifier], share_param_name)
    pr_grad_flat = torch.cat([p.clone().flatten().to(device) for _, p in pr_share_grad.items()])

    # gradients for link prediction
    optimizer.zero_grad()
    loss_link_mean.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(params, args.clip)
    link_share_grad = record_grad([gpt_gnn, classifier], share_param_name)
    link_grad_flat = torch.cat([p.clone().flatten().to(device) for n, p in link_share_grad.items()])

    # gradients for node prediction
    optimizer.zero_grad()
    loss_attr_mean.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(params, args.clip)
    attr_share_grad = record_grad([gpt_gnn, classifier], share_param_name)
    attr_grad_flat = torch.cat([p.clone().flatten().to(device) for n, p in attr_share_grad.items()])

    pr_link_cos = cos_(pr_grad_flat.view(1, -1), link_grad_flat.view(1, -1))
    pr_attr_cos = cos_(pr_grad_flat.view(1, -1), attr_grad_flat.view(1, -1))

    return pr_link_cos, pr_attr_cos

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

def randint():
    return np.random.randint(2**32 - 1)

def node_classification_sample(seed, nodes, time_range):
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = any
    x_ids = np.arange(args.batch_size)
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, graph.y[samp_nodes]

for epoch in np.arange(args.n_epoch) + 1:
    train_data_pr = 
    valid_data_pr = 
    train_data_au = 
 
    '''
            Prepare for Training

            Till here from: for epoch ... ---------------------------------------------------------------------------------------------
            I already created train and other data sets so those will be used instead of the above ones
    '''
    train_losses = []
    train_pr_losses = []
    train_link_losses = []
    train_attr_losses = []
    gpt_gnn.neg_queue_size = args.queue_size * epoch // args.n_epoch
    gpt_gnn.train()
    model_pr.train()
    gpt_gnn_meta.train()
    gnn_meta.train()
    classifier_meta.train()

    pr_link_cos = 0
    pr_attr_cos = 0
    '''
            Train on primary task and auxiliary tasks
    '''
    for i in range(args.n_batch):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = train_data_pr[i]
        data, rem_edge_list, ori_edge_list, attr, (start_idx, end_idx) = train_data_au[i]
        node_feature_au, node_type_au, edge_time_au, edge_index_au, edge_type_au, node_dict_au, edge_dict_au = data
        node_feature_au = node_feature_au.detach()
        # mindegyik feature a kezdeti embeding feature lesz
        node_feature_au[start_idx: end_idx] = init_emb
        
        loss_pr_meta = 0 
        for j in range(args.n_fold):
            # meta model, az eredeti modellek másolata
            gnn_meta.to(device)
            gnn_meta.load_state_dict(gnn.state_dict())
            gpt_gnn_meta.to(device)
            gpt_gnn_meta.load_state_dict(gpt_gnn.state_dict())
            classifier_meta.to(device)
            classifier_meta.load_state_dict(classifier.state_dict())


            # primary task
            node_rep = gnn_meta.forward(node_feature.to(device), node_type.to(device), edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res = classifier_meta.forward(node_rep[x_ids])
            loss_pr = criterion(res, ylabel.to(device))   # loss per example => reduce = false

            # auxiliary task
            node_emb_au = gpt_gnn_meta.gnn(node_feature_au.to(device), node_type_au.to(device), edge_time_au.to(device), \
                                edge_index_au.to(device), edge_type_au.to(device))
            loss_link, _ = gpt_gnn_meta.link_loss(node_emb_au, rem_edge_list, ori_edge_list, node_dict_au, target_type,
                                            use_queue=True, update_queue=True)
         
            loss_attr = gpt_gnn_meta.feat_loss(node_emb_au[start_idx: end_idx], torch.FloatTensor(attr).to(device))

            # for log
            loss_pr_mean = loss_pr.mean()
            loss_link_mean = loss_link.mean()
            loss_attr_mean = loss_attr.mean()
            if not dummy_flag_meta:
                #egyszer megy csak bele
                optimizer_meta.zero_grad()
                loss = loss_pr_mean + loss_link_mean + loss_attr_mean
                loss.backward(retain_graph=True)
                clean_param_name([gpt_gnn_meta, classifier_meta], share_param_name)
                clean_param_name([gpt_gnn_meta, classifier_meta], link_private_param_name)
                clean_param_name([gpt_gnn_meta, classifier_meta], attr_private_param_name)
                dummy_flag_meta = True
            else:
                loss = 0

            pr_link_cos, pr_attr_cos = get_cos(args, device, classifier_meta, params_meta, gpt_gnn_meta, optimizer_meta, share_param_name, cos_, loss_pr_mean, loss_link_mean, loss_attr_mean)

            # embeddings for v-net
            loss_pr_emb = torch.stack((loss_pr, \
                    torch.ones([len(loss_pr)]).to(device), \
                    torch.zeros([len(loss_pr)]).to(device), \
                    torch.zeros([len(loss_pr)]).to(device), \
                    torch.full([len(loss_pr)], 1.0).to(device),
                ))
            loss_link_emb = torch.stack((loss_link, \
                    torch.zeros([len(loss_link)]).to(device), \
                    torch.ones([len(loss_link)]).to(device), \
                    torch.zeros([len(loss_link)]).to(device), \
                    torch.full([len(loss_link)], pr_link_cos.item()).to(device)
                ))
            loss_attr_emb = torch.stack((loss_attr, \
                    torch.zeros([len(loss_attr)]).to(device), \
                    torch.zeros([len(loss_attr)]).to(device), \
                    torch.ones([len(loss_attr)]).to(device), \
                    torch.full([len(loss_attr)], pr_attr_cos.item()).to(device)
                ))

            loss_pr_emb = loss_pr_emb.transpose(1,0)
            loss_link_emb = loss_link_emb.transpose(1,0)
            loss_attr_emb = loss_attr_emb.transpose(1,0)

            # compute weight
            v_pr = vnet(loss_pr_emb)
            v_link = vnet(loss_link_emb)
            v_attr = vnet(loss_attr_emb)

            # compute loss
            loss_pr_avg = (loss_pr * v_pr).mean()
            loss_link_avg = (loss_link * v_link).mean()
            loss_attr_avg = (loss_attr * v_attr).mean()
            loss_meta = loss_pr_avg + loss_link_avg + loss_attr_avg

            # one step update of model parameter (fake) (Eq.6)
            optimizer_meta.zero_grad()
            loss_meta.backward()
            torch.nn.utils.clip_grad_norm_(params_meta, args.clip)
            optimizer_meta.step()
            train_step_meta += 1
            scheduler_meta.step(train_step_meta)

            # primary loss with updated parameter (Eq.7)
            node_rep = gnn_meta.forward(node_feature.to(device), node_type.to(device), edge_time.to(device),
                                        edge_index.to(device), edge_type.to(device))
            res = classifier_meta.forward(node_rep[x_ids])
            loss_pr_meta += criterion(res, ylabel.to(device)).mean() #scalar

        # in each batch

        # backward and update v-net params (Eq.9)
        optimizer_v.zero_grad()
        loss_pr_meta.backward()
        optimizer_v.step()

        # with the updated weight, update model parameters (true) (Eq.8)
        # primary task
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = train_data_pr[i]
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), edge_time.to(device),
                               edge_index.to(device), edge_type.to(device))
        res = classifier.forward(node_rep[x_ids])
        loss_pr = criterion(res, ylabel.to(device))   # loss per example
        # auxiliary task
        data, rem_edge_list, ori_edge_list, attr, (start_idx, end_idx) = train_data_au[i]
        node_feature_au, node_type_au, edge_time_au, edge_index_au, edge_type_au, node_dict_au, edge_dict_au = data
        node_feature_au = node_feature_au.detach()
        node_feature_au[start_idx: end_idx] = init_emb
        node_emb_au = gpt_gnn.gnn(node_feature_au.to(device), node_type_au.to(device), edge_time_au.to(device), \
                                  edge_index_au.to(device), edge_type_au.to(device))
        loss_link, _ = gpt_gnn.link_loss(node_emb_au, rem_edge_list, ori_edge_list, node_dict_au, target_type,
                                         use_queue=True, update_queue=True)
        loss_attr = gpt_gnn.feat_loss(node_emb_au[start_idx: end_idx], torch.FloatTensor(attr).to(device))

        # for log
        loss_pr_mean = loss_pr.mean()
        loss_link_mean = loss_link.mean()
        loss_attr_mean = loss_attr.mean()
        if not dummy_flag:
            optimizer.zero_grad()
            loss = loss_pr_mean + loss_link_mean + loss_attr_mean
            loss.backward(retain_graph=True)
            clean_param_name([gpt_gnn, classifier], share_param_name)
            clean_param_name([gpt_gnn, classifier], link_private_param_name)
            clean_param_name([gpt_gnn, classifier], attr_private_param_name)
            dummy_flag = True

        pr_link_cos, pr_attr_cos = get_cos(args, device, classifier, params, gpt_gnn, optimizer, share_param_name, cos_, loss_pr_mean, loss_link_mean, loss_attr_mean)

        print('link cos: %f\tattr cos: %f'%(pr_link_cos.item(), pr_attr_cos.item()))

        # embeddings for v-net
        # shape = (5, batch size)
        loss_pr_emb = torch.stack((loss_pr, \
                torch.ones([len(loss_pr)]).to(device), \
                torch.zeros([len(loss_pr)]).to(device), \
                torch.zeros([len(loss_pr)]).to(device), \
                torch.full([len(loss_pr)], 1.0).to(device),
            ))
        loss_link_emb = torch.stack((loss_link, \
                torch.zeros([len(loss_link)]).to(device), \
                torch.ones([len(loss_link)]).to(device), \
                torch.zeros([len(loss_link)]).to(device), \
                torch.full([len(loss_link)], pr_link_cos.item()).to(device)
            ))
        loss_attr_emb = torch.stack((loss_attr, \
                torch.zeros([len(loss_attr)]).to(device), \
                torch.zeros([len(loss_attr)]).to(device), \
                torch.ones([len(loss_attr)]).to(device), \
                torch.full([len(loss_attr)], pr_attr_cos.item()).to(device)
            ))
        # embeddings for v-net
        #shape = (batch size, 5)
        loss_pr_emb = loss_pr_emb.transpose(1, 0)
        loss_link_emb = loss_link_emb.transpose(1, 0)
        loss_attr_emb = loss_attr_emb.transpose(1, 0)

        # compute weight
        with torch.no_grad():
            v_pr = vnet(loss_pr_emb)
            v_link = vnet(loss_link_emb)
            v_attr = vnet(loss_attr_emb)

        # compute loss
        loss_pr_avg = (loss_pr * v_pr).mean()
        loss_link_avg = (loss_link * v_link).mean()
        loss_attr_avg = (loss_attr * v_attr).mean()
        loss_pr_avg_weighted = loss_pr.mean()
        loss_link_avg_weighted = loss_link.mean()
        loss_attr_avg_weighted = loss_attr.mean()

        print((
                  "Epoch: %d  Batch: %d  Train Loss Pr: %.2f  Train Loss Link: %.2f  Train Loss Attr: %.2f  Pr_Weight_Mean: %.4f Link_Weight_Mean: %.4f Attr_Weight_Mean: %.4f Pr_Weight_Std: %.4f Link_Weight_Std: %.4f Attr_Weight_Std: %.4f ") %
              (epoch, i, loss_pr_avg_weighted, loss_link_avg_weighted, loss_attr_avg_weighted, v_pr.mean().item(), v_link.mean().item(),
               v_attr.mean().item(), v_pr.std().item(), v_link.std().item(), v_attr.std().item()))

        # total loss
        loss = loss_pr_avg + loss_link_avg + loss_attr_avg
        train_pr_losses += [loss_pr_avg_weighted.cpu().detach().tolist()]
        train_link_losses += [loss_link_avg_weighted.item()]
        train_attr_losses += [loss_attr_avg_weighted.item()]

        # optimize model parameters
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        del res, loss_pr, loss_attr, loss_link, loss_pr_emb, loss_link_emb, loss_attr_emb, v_pr, v_link, v_attr
        del loss_pr_mean, loss_attr_mean, loss_link_mean, loss_pr_avg_weighted, loss_link_avg_weighted, loss_attr_avg_weighted


    '''
        Valid with only primary task (2017 <= time <= 2017)

        in each epoch
    '''
    model_pr.eval()
    with torch.no_grad():
        valid_loss_pr_mean = 0
        valid_loss_link_mean = 0
        valid_loss_attr_mean = 0
        valid_res = []
        loss = 0
        for i in range(args.valid_batch):
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data_pr[i]
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                    edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res  = classifier.forward(node_rep[x_ids])
            loss += criterion(res, ylabel.to(device)).mean() / args.valid_batch #scalar
            # loss = criterion(res, ylabel.to(device))
        
            '''
                Calculate Valid F1. Update the best model based on highest F1 score.
            '''
            valid_res += [f1_score(res.argmax(dim=1).cpu().tolist(), ylabel.tolist(), average='micro')]
        
        valid_f1 = np.average(valid_res)
        if valid_f1 > best_val:
            best_val = valid_f1
            torch.save(model_pr,
                    os.path.join(args.model_dir, args.task_name + '_' + args.conv_name + '_' + args.model_add_name))
            print('UPDATE!!!')
        
        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid F1: %.4f") % \
              (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_pr_losses), \
                    loss.cpu().detach().tolist(), valid_f1))
        del res, loss
    del train_data_pr, valid_data_pr, train_data_au


    '''
        Test 
    '''
    if epoch > 0 and epoch % 5 == 0:
        best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name + '_' + args.model_add_name), map_location=CUDA_STR).to(device)
        best_model.eval()
        gnn_best, classifier_best = best_model
        with torch.no_grad():
            test_res = []
            for _ in range(10):
                node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = ... # teszt gráf
                paper_rep = gnn_best.forward(node_feature.to(device), node_type.to(device), \
                            edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
                res = classifier_best.forward(paper_rep)
                test_f1 = f1_score(res.argmax(dim=1).cpu().tolist(), ylabel.tolist(), average='micro')
                test_res += [test_f1]
        print('Best Test F1: %.4f' % np.average(test_res))

