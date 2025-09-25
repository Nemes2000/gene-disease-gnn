import torch

class Config():
    #data creataion params
    # If you use the test dataset set min_disease_s_gene_number to 0, and train test_split to 0.5 
    min_disease_s_gene_number = 10
    min_gene_s_disease_number = 3
    test_size = 0.2
    val_size = 0.0
    train_test_split = 0.2
    test_val_split = 1
    process_files = False
    pos_class_weight = 1
    disease_idx = None

    # Test dataset param
    test_dataset = False

    #train params
    learning_rate = 0.03682
    weight_decay = 0.00198
    optimalization_step = 1
    optimizer = torch.optim.AdamW
    optimizer_map = {
        'adam': torch.optim.Adam,
        'adamW': torch.optim.AdamW
    }
    epochs = 1

    #layer params
    dropout_rate = 0.041454
    num_classes = 2
    hidden_channels = 39
    num_layers = 1
    in_channels = 0 # will receave from dataset
    out_channels = 0 # will receave from dataset
    
    #GPU params
    avail_gpus = min(1, torch.cuda.device_count())

    #folder params
    checkpoint_path = "../data/saved_models/"
    raw_data_path = "../../data/raw/"
    processed_data_dir = "../../data/processed"
    model_name = "basic"
    sweep_num = 0

    #wwandb params
    wandb_api_key = "e1f878235d3945d4141f9f8e5af41d712fca6eba"
    wandb_project_name = "gnn_multitask"

    #Multitask learning params
    pr_disease_idx = None
    aux_disease_idxs = []
    pr_pos_class_weight = 1
    aux_pos_class_weights = []
    aux_task_num = 0
    pretrain_epochs = 5
    v_act_type = "sigmoid"
    clip = 0.5
    v_emb_dim = 2
    v_dropout_rate = 0.041454
    v_lr = 0.005
    v_wd = 0.004870443319486102
    mt_lr = 0.01859073883306547
    mt_wd = 0.004309004534608284
    mt_eps = 8.974635961426902e-05
    mt_hidden_1 = 5
    mt_hidden_2 = 33

    def set_train_val_test_dataset_size(self, test_size, val_size):
        self.train_test_split = val_size + test_size
        self.test_val_split = test_size / (test_size + val_size)


from enum import Enum
class ModelTypes(str, Enum):
    BASIC = "basic"
    CLS_WEIGHT = "cls_weight"
    MULTITASK = "multitask"
