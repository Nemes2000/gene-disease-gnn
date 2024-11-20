import wandb
import torch_geometric.loader as geom_data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import os

from models.lightning_gnn_model import LightningGNNModel
from config import Config



def train_node_classifier(model_name, dataset, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, num_workers=11, persistent_workers=True)

    # Create a PyTorch Lightning trainer
    root_dir = os.path.join(Config.checkpoint_path, "TestGCN" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        accelerator="auto",
        devices=1,
        max_epochs=Config.epochs,
        enable_progress_bar=False,
        log_every_n_steps=1,
        logger=pl.loggers.WandbLogger(project="gene-disease-test", log_model="all")
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, f"TestGCN{model_name}.ckpt")
    # if os.path.isfile(pretrained_filename):
    #     print("Found pretrained model, loading...")
    #     model = TestGCN.load_from_checkpoint(pretrained_filename)
    # else:
    pl.seed_everything()
    model = LightningGNNModel(
        model_name=model_name, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs
    )
    trainer.fit(model, node_data_loader, node_data_loader)
    model = LightningGNNModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, dataloaders=node_data_loader)
    result = {"test": test_result[0]["test_acc"]}
    wandb.finish()
    return model, result
      
      