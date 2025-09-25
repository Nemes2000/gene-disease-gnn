from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score, ConfusionMatrixDisplay, roc_curve

from models.basic_model import BasicGNNModel
from config import Config, ModelTypes
from models.multitask_model import MultiTaskGNNModel
from models.utils import clean_param_name, get_cos
from models.weight_model import Weight

class LightningGNNModel(pl.LightningModule):
    def __init__(self, model_name):
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
            self.mt_gnn = MultiTaskGNNModel(gnn=BasicGNNModel(), aux_tasks_num=Config.aux_task_num)
            self.mt_gnn_meta = MultiTaskGNNModel(gnn=BasicGNNModel(), aux_tasks_num=Config.aux_task_num)
            self.cos_ = torch.nn.CosineSimilarity()
            
            # weigth model, optimizer_v is not saved in the checkpoint !!!
            self.vnet = Weight(4, Config.weight_emb_dim, 1, Config.weigth_act_type)
            self.optimizer_v = torch.optim.Adam(self.vnet.parameters(), lr=Config.weight_lr, weight_decay=1e-3)

            # optimizer and scheduler are passed back with configure_optimizers => will be saved with the modell
            self.params = self.mt_gnn.parameters()
            self.optimizer = torch.optim.AdamW(self.params, lr=Config.learning_rate, weight_decay=Config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max= Config.epochs, eta_min=1e-6)

            # wont be saved in the checkpoint
            self.params_meta = self.mt_gnn_meta.parameters()
            self.optimizer_meta = torch.optim.AdamW(self.params_meta, lr=Config.learning_rate, weight_decay=Config.weight_decay)
            self.scheduler_meta = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_meta, T_max=Config.epochs, eta_min=1e-6)
            
            self.unused_params_cleared = False
            self.clear_unused_meta_params = False
            self.train_step_meta = 0
            self.aux_cos_df = pd.DataFrame(columns=["epoch", "aux_idx", "cos", "weight"])
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
        loss_pr_meta = 0 
        pr_aux_s_cos = 0
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
        _loss_pr = loss_pr.unsqueeze(0)
        loss_pr_emb = torch.stack((
            _loss_pr,
            torch.ones_like(_loss_pr),
            torch.zeros_like(_loss_pr),
            torch.full_like(_loss_pr, 1.0),
        )).transpose(1,0)

        
        loss_aux_s_emb = []
        for idx, loss_aux in enumerate(loss_aux_s):
            loss_aux = loss_aux.unsqueeze(0)  # shape: [1]

            emb = torch.stack([
                loss_aux,
                torch.zeros_like(loss_aux),
                torch.ones_like(loss_aux),
                torch.full_like(loss_aux, pr_aux_s_cos[idx].item())
            ], dim=0)  # shape: [5, N]

            # shape: [N, 5]
            emb = emb.transpose(1, 0)
            loss_aux_s_emb.append(emb)

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
        self.train_step_meta += 1
        self.scheduler_meta.step(self.train_step_meta)

        # primary loss with updated parameter (Eq.7)
        _loss_pr_meta, _ = self.mt_gnn_meta.forward(data, mode)
        loss_pr_meta += _loss_pr_meta

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
            self.optimizer.zero_grad()
            loss = loss_pr_mean + sum(loss_aux_s_mean)
            loss.backward(retain_graph=True)
            clean_param_name([self.mt_gnn], self.share_param_name)
            clean_param_name([self.mt_gnn], self.private_param_name)
            self.unused_params_cleared = True

        pr_aux_s_cos = get_cos(self.params, self.mt_gnn, self.optimizer, self.share_param_name, self.cos_, loss_pr_mean, loss_aux_s_mean)

        # embeddings for v-net
        # shape = (5, batch size)   
        _loss_pr = loss_pr.unsqueeze(0)
        loss_pr_emb = torch.stack((
            _loss_pr,
            torch.ones_like(_loss_pr),
            torch.zeros_like(_loss_pr),
            torch.full_like(_loss_pr, 1.0),
        ))
            
        loss_aux_s_emb = []
        for idx, loss_aux in enumerate(loss_aux_s):
            loss_aux = loss_aux.unsqueeze(0)  # shape: [1]

            emb = torch.stack([
                loss_aux,
                torch.zeros_like(loss_aux),
                torch.ones_like(loss_aux),
                torch.full_like(loss_aux, pr_aux_s_cos[idx].item())
            ], dim=0)  # shape: [5, N]

            # shape: [N, 5]
            emb = emb.transpose(1, 0)
            loss_aux_s_emb.append(emb)
        
        # embeddings for v-net
        #shape = (batch size, 5)
        loss_pr_emb = loss_pr_emb.transpose(1, 0)

        # compute weight
        with torch.no_grad():
            v_pr = self.vnet(loss_pr_emb)
            v_aux_s = [self.vnet(loss_aux_emb) for loss_aux_emb in loss_aux_s_emb]

        for c, idx, w in zip(pr_aux_s_cos, Config.aux_disease_idxs, v_aux_s):
            self.aux_cos_df = pd.concat(
                [self.aux_cos_df, pd.DataFrame([{
                    "epoch": self.current_epoch,
                    "aux_idx": idx,
                    "cos": c.item(),
                    "weight": w.item()
                }])],
                ignore_index=True
            )

        self.aux_cos_df = pd.concat(
                [self.aux_cos_df, pd.DataFrame([{
                    "epoch": self.current_epoch,
                    "aux_idx": Config.pr_disease_idx,
                    "cos": 1,
                    "weight": v_pr.item()
                }])],
                ignore_index=True
            )
           
        self.aux_cos_df.to_csv(f"results/multitask/{Config.pr_disease_idx}_aux_cosine_epoch.csv", index=False)

        # compute loss
        loss_pr_avg = (loss_pr * v_pr).mean()
        loss_aux_avg_s = [(loss_aux * v_aux).mean() for loss_aux, v_aux in zip(loss_aux_s, v_aux_s)]
        loss_pr_avg_weighted = loss_pr.mean()
        loss_aux_avg_weighted = torch.stack([l.mean() for l in loss_aux_s]).mean()

        v_aux_s_tensor = torch.tensor(v_aux_s)
        v_aux_s_mean = v_aux_s_tensor.mean().item()
        v_aux_s_std = v_aux_s_tensor.std().item()
        print(("Train Loss Pr: %.2f  Train Loss Aux: %.2f  Pr_Weight_Mean: %.4f Aux_Weight_Mean: %.4f Aux_Weight_Std: %.4f ") %
                (loss_pr_avg_weighted, loss_aux_avg_weighted, v_pr.mean().item(), v_aux_s_mean, v_aux_s_std))

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

        return loss
    
    def get_model_params(self):
        self.private_param_name = [
            name for name, param in self.mt_gnn.named_parameters()
            if ('aux_layers' in name or 'pr_layer' in name) and param.requires_grad
        ]

        self.share_param_name = []
        for n, p in self.mt_gnn.named_parameters():
            if n not in self.private_param_name and p.requires_grad:
                self.share_param_name.append(n)

    def training_step(self, data):
        if self.model_name == ModelTypes.MULTITASK:
            if self.current_epoch < Config.pretrain_epochs:
                loss = self.mt_gnn.forward(data, mode="train", is_pretrain=True)
            else: 
                loss = self.multitask_forward(data)
        else:
            loss = self.basic_forward(data, mode="train")

        return loss

    def validation_step(self, data):
        if self.model_name == ModelTypes.MULTITASK:
            pr_loss, _ = self.mt_gnn(data, mode="val")
            self.log("val_loss", pr_loss)
            return pr_loss
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
        pr_loss, embeding = self.mt_gnn(data, mode="test")

        print("Creating pr disease statisctic...")
        
        y_masked = data.y[:, Config.pr_disease_idx].cpu()

        y_true = data.y[:, Config.pr_disease_idx].detach().cpu().numpy()
        y_pred = (embeding[:, Config.pr_disease_idx] > 0.5).int().detach().cpu().numpy()

        #tn, fp, fn, tp
        cm = confusion_matrix(y_true, y_pred)

        # y_pred = torch.where(x_pred_masked > 0.5, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))
        # cm = confusion_matrix(y_masked, y_pred)

        roc_auc = 0
        if len(np.unique(y_masked)) > 1:
            roc_auc = roc_auc_score(y_masked, y_pred)

            fpr, tpr, thresholds = roc_curve(y_masked, y_pred)

            # Ábra készítés
            plt.figure()
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")

            # Mentés képként
            plt.savefig(f"results/multitask/{Config.pr_disease_idx}_roc_curve.png", dpi=300)
            plt.close()
        
        df = pd.DataFrame({"disease_idx": [Config.pr_disease_idx], 
                           "acc": [accuracy_score(y_masked, y_pred)], 
                           "f1": [f1_score(y_masked, y_pred)], 
                           "recal": [recall_score(y_masked, y_pred)], 
                           "precision": [precision_score(y_masked, y_pred)], 
                           "roc-auc":[roc_auc],
                           "auprc": [average_precision_score(y_masked, y_pred)], 
                           "cm": [" ".join(map(str, cm.flatten()))], 
                           "x_sum": [ y_pred.sum().item()], 
                           "y_sum": [y_masked.sum().item()]})
            
        df.to_csv(f"results/multitask/{Config.pr_disease_idx}_classification.csv", index=False, sep=",")
        print("Creating pr disease statisctic. DONE")

        return pr_loss

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

        df = pd.DataFrame({"disease_idx": [], "acc": [], "f1": [], "recal": [], "precision": [], "roc-auc":[],"auprc": [], "cm": [], "x_sum": [], "y_sum": []})
        print("Creating pred disease statisctic...")
        for idx in range(data.y.shape[1]):
            x_pred_masked = x_pred[:, idx]
            y_masked = data.y[:, idx]

            y_pred_binary = torch.where(x_pred_masked > 0.5, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))
            cm = confusion_matrix(y_masked, y_pred_binary)

            roc_auc = 0
            if len(np.unique(y_masked)) > 1:
                roc_auc = roc_auc_score(y_masked, y_pred_binary)
                        
            df.loc[len(df)] = {"disease_idx": idx, 
                               "acc": accuracy_score(y_masked, y_pred_binary), 
                               "f1": f1_score(y_masked, y_pred_binary), 
                               "recal": recall_score(y_masked, y_pred_binary), 
                               "precision": precision_score(y_masked, y_pred_binary),
                               "roc-auc": roc_auc,
                               "auprc": average_precision_score(y_masked, y_pred_binary),
                               "cm": " ".join(map(str, cm.flatten())),
                               "x_sum": y_pred_binary.sum().item(),
                               "y_sum": y_masked.sum().item()}
            
        df.to_csv("results/disease_classifications.csv", index=False, sep=",")
        print("Creating pred disease statisctic. DONE")
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

