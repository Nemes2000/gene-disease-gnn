import statistics
import torch

import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score, ConfusionMatrixDisplay

from models.basic_model import BasicGNNModel
from config import Config, ModelTypes
from models.multitask_model import MultiTaskGNNModel
from models.utils import clean_param_name, get_cos
from models.weight_model import Weight

class LightningGNNModel(pl.LightningModule):
    def __init__(self, model_name, pr_task: str, aux_tasks: list[str]):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name

        if model_name == ModelTypes.BASIC:
            self.loss_function = torch.nn.BCEWithLogitsLoss()
            self.model = BasicGNNModel()
        elif model_name == ModelTypes.CLS_WEIGHT:
            pos_weight=torch.tensor(Config.pos_class_weight)
            self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)#FOR every disease --> 
            self.model = BasicGNNModel()
        elif model_name == ModelTypes.MULTITASK:
            self.mt_gnn = MultiTaskGNNModel()
            self.mt_gnn_meta = MultiTaskGNNModel()
            self.cos_ = torch.nn.CosineSimilarity()
            
            # weigth model, optimizer_v is not saved in the checkpoint !!!
            self.vnet = Weight(5, Config.weight_emb_dim, 1, Config.weigth_act_type)
            self.optimizer_v = torch.optim.Adam(self.vnet.parameters(), lr=Config.weight_lr, weight_decay=1e-3)

            # optimizer and scheduler are passed back with configure_optimizers => will be saved with the modell
            self.params = self.mt_gnn.parameters()
            self.optimizer = torch.optim.AdamW(self.params, weight_decay = 1e-2, eps=1e-06, lr = Config.max_lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max= Config.epochs, eta_min=1e-6)
            self.scheduler_meta = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_meta, T_max=Config.epochs, eta_min=1e-6)

            # wont be saved in the checkpoint
            self.params_meta = self.mt_gnn_meta.parameters()
            self.optimizer_meta = torch.optim.AdamW(self.params_meta, weight_decay = 1e-2, eps=1e-06, lr = Config.max_lr)
            
            self.unused_params_cleared = False
            self.clear_unused_meta_params = False
            self.get_model_params()

    def basic_forward(self, data, mode="train"):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x_pred = self.model.forward(x, edge_index, edge_weight)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.test_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"
        
        if Config.disease_idx:
            mask_t = mask.clone()
            mask_t.zero_()
            mask_t[:, Config.disease_idx] = mask[:, Config.disease_idx]
            mask = mask_t

        # After masking out the given percentage of the matrix i cant recreate the original one with same shape
        # disease_num = data.y.shape[1]
        # x_pred[mask].reshape(-1, disease_num)

        loss = self.loss_function(x_pred[mask].unsqueeze(1), data.y[mask].unsqueeze(1))

        x_pred_binary = torch.where(x_pred[mask] > 0.5, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))
        y_masked = data.y[mask]
        acc = accuracy_score(y_masked, x_pred_binary)

        if mode == "test":
            return loss, acc, x_pred
        return loss, acc
    
    def multitask_forward(self, data, mode = "train"):
        #TODO: set all feature to a start one => i dont need that ?
        #node_feature_au = init_emb
        
        loss_pr_meta = 0 
        pr_aux_s_cos = 0
        for _ in range(Config.n_fold):
            # meta models are the copy of original ones
            self.mt_gnn_meta.load_state_dict(self.mt_gnn.state_dict())

            # loss per example => reduce = false
            loss_pr, loss_aux_s = self.mt_gnn_meta.forward(data, mode)

            # for log
            loss_pr_mean = loss_pr.mean()
            loss_aux_s_mean = [loss_aux.mean() for loss_aux in loss_aux_s]


            # clears the unused meta params
            if not self.clear_unused_meta_params:
                self.optimizer_meta.zero_grad()
                loss_meta = loss_pr_mean + sum(loss_aux_s_mean)
                loss_meta.backward(retain_graph=True)
                clean_param_name([self.mt_gnn_meta], self.share_param_name)
                clean_param_name([self.mt_gnn_meta] , self.private_param_name)
                self.clear_unused_meta_params = True

            pr_aux_s_cos = get_cos(self.params_meta, self.mt_gnn_meta, self.optimizer_meta, self.share_param_name, self.cos_, loss_pr_mean, loss_aux_s_mean)

            # embeddings for v-net
            loss_pr_emb = torch.stack((loss_pr, \
                    torch.ones([len(loss_pr)], device=loss_pr.device), \
                    torch.zeros([len(loss_pr)], device=loss_pr.device), \
                    torch.zeros([len(loss_pr)], device=loss_pr.device), \
                    torch.full([len(loss_pr)], 1.0, device=loss_pr.device),
                ))
            
            loss_aux_s_emb = [torch.stack((loss_aux, \
                    torch.zeros([len(loss_aux)], device=loss_pr.device), \
                    torch.zeros([len(loss_aux)], device=loss_pr.device), \
                    torch.ones([len(loss_aux)], device=loss_pr.device), \
                    torch.full([len(loss_aux)], pr_aux_s_cos[idx].item(), device=loss_pr.device)
                )).transpose(1,0)
                for idx, loss_aux in enumerate(loss_aux_s)]

            loss_pr_emb = loss_pr_emb.transpose(1,0)

            # compute weight
            v_pr = self.vnet(loss_pr_emb)
            v_aux_s = [self.vnet(loss_aux_emb) for loss_aux_emb in loss_aux_s_emb]

            # compute loss
            loss_pr_avg = (loss_pr * v_pr).mean()
            loss_aux_avg_s = [(loss_aux * v_aux).mean() for loss_aux, v_aux in zip(loss_aux_s, v_aux_s)]
            loss_meta = loss_pr_avg  + sum(loss_aux_avg_s)

            # one step update of model parameter (fake) (Eq.6)
            self.optimizer_meta.zero_grad()
            loss_meta.backward()
            torch.nn.utils.clip_grad_norm_(self.params_meta, Config.clip)
            self.optimizer_meta.step()
            train_step_meta += 1
            self.scheduler_meta.step(train_step_meta)

            # primary loss with updated parameter (Eq.7)
            loss_pr_meta += self.mt_gnn_meta.forward(data, mode)

        # in each batch

        # backward and update v-net params (Eq.9)
        self.optimizer_v.zero_grad()
        loss_pr_meta.backward()
        self.optimizer_v.step()

        # with the updated weight, update model parameters (true) (Eq.8)
        loss_pr, loss_aux_s = self.mt_gnn.forward(data, mode)

        # for log
        loss_pr_mean = loss_pr.mean()
        loss_aux_s_mean = [loss_aux.mean() for loss_aux in loss_aux_s]
        if not self.unused_params_cleared:
            loss = loss_pr_mean + sum(loss_aux_s_mean)
            loss.backward(retain_graph=True)
            clean_param_name([self.mt_gnn], self.share_param_name)
            clean_param_name([self.mt_gnn], self.private_param_name)
            self.optimizer.zero_grad()
            self.unused_params_cleared = True

        pr_aux_s_cos = get_cos(self.params, self.mt_gnn, self.optimizer, self.share_param_name, self.cos_, loss_pr_mean, loss_aux_s_mean)

        for c in pr_aux_s_cos:
            print('aux cos: %f'%(c.item()))

        # embeddings for v-net
        # shape = (5, batch size)   
        loss_pr_emb = torch.stack((loss_pr, \
                    torch.ones([len(loss_pr)], device=loss_pr.device), \
                    torch.zeros([len(loss_pr)], device=loss_pr.device), \
                    torch.zeros([len(loss_pr)], device=loss_pr.device), \
                    torch.full([len(loss_pr)], 1.0, device=loss_pr.device),
                ))
        loss_aux_s_emb = [torch.stack((loss_aux, \
                    torch.zeros([len(loss_aux)], device=loss_pr.device), \
                    torch.zeros([len(loss_aux)], device=loss_pr.device), \
                    torch.ones([len(loss_aux)], device=loss_pr.device), \
                    torch.full([len(loss_aux)], pr_aux_s_cos[idx].item(), device=loss_pr.device)
                )).transpose(1,0)
                for idx, loss_aux in enumerate(loss_aux_s)]
        
        # embeddings for v-net
        #shape = (batch size, 5)
        loss_pr_emb = loss_pr_emb.transpose(1, 0)

        # compute weight
        with torch.no_grad():
            v_pr = self.vnet(loss_pr_emb)
            v_aux_s = [self.vnet(loss_aux_emb) for loss_aux_emb in loss_aux_s_emb]

        # compute loss
        loss_pr_avg = (loss_pr * v_pr).mean()
        loss_aux_avg_s = [(loss_aux * v_aux).mean() for loss_aux, v_aux in zip(loss_aux_s, v_aux_s)]
        loss_pr_avg_weighted = loss_pr.mean()
        loss_aux_avg_weighted = statistics.mean(loss_aux_s.mean())

        print(("Train Loss Pr: %.2f  Train Loss Aux: %.2f  Pr_Weight_Mean: %.4f Aux_Weight_Mean: %.4f Pr_Weight_Std: %.4f Aux_Weight_Std: %.4f ") %
                (loss_pr_avg_weighted, loss_aux_avg_weighted, v_pr.mean().item(), v_aux_s.mean().item(), v_pr.std().item(), v_aux_s.std().item()))

        loss = loss_pr_avg + sum(loss_aux_avg_s)
        # train_pr_losses += [loss_pr_avg_weighted.cpu().detach().tolist()]
        # train_aux_losses += [loss_aux_avg_weighted.cpu().detach().tolist()]

        # optimize model parameters => lightning does it automaticly
        # self.optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.params, Config.clip)
        # self.optimizer.step()
        # self.train_step += 1
        # self.scheduler.step(self.train_step)

        return loss, acc, x_pred
    
    def get_model_params(self):
        self.private_param_name = [p[0] for p in self.mt_gnn.named_parameters() if 'aux_classifiers' in p[0] and p[1].requires_grad]
        self.share_param_name = []

        #TODO: most a shared-be benne van a pr taskhoz tartozó classifier is, miért????????????? => eredetileg is
        for n, p in self.mt_gnn.named_parameters():
            if n not in self.private_param_name and p.requires_grad:
                self.share_param_name.append(n)

    def training_step(self, data):
        if self.model_name == ModelTypes.MULTITASK:
            if self.current_epoch < Config.pretrain_epochs:
                loss = self.mt_gnn.forward(data, mode="train", is_pretrain=True)
            else: 
                loss, acc = self.multitask_forward(data)
        else:
            loss, acc = self.basic_forward(data, mode="train")

        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, data):
        if self.model_name == ModelTypes.MULTITASK:
            loss, acc = self.multitask_forward(data, mode="val")
        else:
            loss, acc = self.basic_forward(data, mode="val")

        self.log("val_acc", acc)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, data):
        if self.model_name == ModelTypes.BASIC:
            return self.basic_test_step(data)
        elif self.model_name == ModelTypes.CLS_WEIGHT:
            return self.cls_test_step(data)
        elif self.model_name == ModelTypes.MULTITASK:
            return self.multitask_test_step(data)
        
    def multitask_test_step(self, data):
        loss, acc, x_pred = self.multitask_forward(data, mode="test")
        return _

    def cls_test_step(self, data):
        loss, acc, x_pred = self.basic_forward(data, mode="test")

        if Config.disease_idx:
            x_pred_masked = x_pred[:, Config.disease_idx]
            y_masked = data.y[:, Config.disease_idx]
        else:
            x_pred_masked = x_pred[data.test_mask]
            y_masked = data.y[data.test_mask]

        x_pred_binary = torch.where(x_pred_masked > 0.5, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))

        self.log('test_loss', loss)
        if len(np.unique(y_masked)) > 1:
            self.log("ROC-AUC", roc_auc_score(y_masked, x_pred_binary))

        self.log("F1 score", f1_score(y_masked, x_pred_binary))
        self.log("Accuracy", acc)
        self.log("Recall", recall_score(y_masked, x_pred_binary))
        self.log("Precision", precision_score(y_masked, x_pred_binary))
        self.log("AUPRC", average_precision_score(y_masked, x_pred_binary))

        cm = confusion_matrix(y_masked, x_pred_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot().figure_.savefig(f'results/matrices/confusion_matrix_{Config.disease_idx}.png')
        with open(f'results/matrices/confusion_matrix_{Config.disease_idx}.txt', "w") as file:
            file.write(f"""acc: {accuracy_score(y_masked, x_pred_binary)}, 
f1: {f1_score(y_masked, x_pred_binary)}, 
recal: {recall_score(y_masked, x_pred_binary)}, 
precision: {precision_score(y_masked, x_pred_binary)},
auc: {roc_auc_score(y_masked, x_pred_binary)},
auprc: {average_precision_score(y_masked, x_pred_binary)},
cm: {" ".join(map(str, cm.flatten()))},
x_sum: {x_pred_binary.sum().item()},
y_sum: {y_masked.sum().item()}""")

        return loss

    def basic_test_step(self,data):
        loss, _, x_pred = self.basic_forward(data, mode="test")

        df = pd.DataFrame({"disease_idx": [], "acc": [], "f1": [], "recal": [], "precision": [], "auc":[],"auprc": [], "cm": [], "x_sum": [], "y_sum": []})
        print("Creating ped disease statisctic...")
        for idx in range(data.y.shape[1]):
            x_pred_masked = x_pred[:, idx]
            y_masked = data.y[:, idx]

            x_pred_binary = torch.where(x_pred_masked > 0.5, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))
            cm = confusion_matrix(y_masked, x_pred_binary)

            auc = 0
            if len(np.unique(y_masked)) > 1:
                auc = roc_auc_score(y_masked, x_pred_binary)
            
            df.loc[len(df)] = {"disease_idx": idx, 
                               "acc": accuracy_score(y_masked, x_pred_binary), 
                               "f1": f1_score(y_masked, x_pred_binary), 
                               "recal": recall_score(y_masked, x_pred_binary), 
                               "precision": precision_score(y_masked, x_pred_binary),
                               "auc": auc,
                               "auprc": average_precision_score(y_masked, x_pred_binary),
                               "cm": " ".join(map(str, cm.flatten())),
                               "x_sum": x_pred_binary.sum().item(),
                               "y_sum": y_masked.sum().item()}
            
        df.to_csv("results/disease_classifications.csv", index=False, sep=",")
        print("Creating ped disease statisctic. DONE")
        return loss

    def configure_optimizers(self):
        if self.model_name in [ModelTypes.BASIC, ModelTypes.CLS_WEIGHT]:
            return Config.optimizer(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        elif self.model_name == ModelTypes.MULTITASK:
            # A scheduler visszaadása dictionary formában, hogy step-et automatikusan hívja a Lightning
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "epoch",    # "step" vagy "epoch", mikor hívja a scheduler.step()
                    "frequency": 1,         # Milyen gyakran (hányadik epoch után) hívja a scheduler.step()}
                }
            }

