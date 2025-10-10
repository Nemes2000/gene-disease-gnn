import torch

from config import Config

def get_cos(params, mt_gnn, optimizer, share_param_name, cos_, loss_pr_mean, loss_aux_s_mean):
    optimizer.zero_grad()
    loss_pr_mean.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(params, Config.clip)
    pr_share_grad = record_grad([mt_gnn], share_param_name)
    pr_grad_flat = torch.cat([p.clone().flatten() for _, p in pr_share_grad.items()])

    pr_aux_s_cos = []

    for loss_aux_mean in loss_aux_s_mean:
        optimizer.zero_grad()
        loss_aux_mean.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(params, Config.clip)
        aux_share_grad = record_grad([mt_gnn], share_param_name)
        aux_grad_flat = torch.cat([p.clone().flatten() for n, p in aux_share_grad.items()])

        pr_aux_cos = cos_(pr_grad_flat.view(1, -1), aux_grad_flat.view(1, -1))
        pr_aux_s_cos.append(pr_aux_cos)

    return pr_aux_s_cos

def clean_param_name(model_list, name):
    '''
        clean param.grad is None in name(unused param in model)
    '''
    for model in model_list:
        for n, v in zip(model.state_dict(), model.parameters()):
            if v.grad is None and n in name:
                name.remove(n)

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