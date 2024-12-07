import torch

class Config():
    #data creataion params
    # If you use the test dataset set min_disease_s_gene_number to 0, and train test_split to 0.5 
    min_disease_s_gene_number = 7
    train_test_split = 0.2
    test_val_split = 0.5
    process_files = False

    # Test dataset param
    test_dataset = False

    #train params
    learning_rate = 0.01
    weight_decay = 5e-4
    optimalization_step = 1
    optimizer = torch.optim.Adam
    optimizer_map = {
        'adam': torch.optim.Adam,
        'adamW': torch.optim.AdamW
    }
    epochs = 1

    #layer params
    dropout_rate = 0.2
    num_classes = 2
    hidden_channels = 16
    num_layers = 2
    in_channels = 0 # will receave from dataset
    out_channels = 0 # will receave from dataset
    
    #GPU params
    avail_gpus = min(1, torch.cuda.device_count())

    #folder params
    checkpoint_path = "../data/saved_models/"
    raw_data_path = "../../data/raw/"
    processed_data_dir = "../../data/processed"
    model_name = "basic"

    #wwandb params
    wandb_api_key = "e1f878235d3945d4141f9f8e5af41d712fca6eba"
    wandb_project_name = "gnn_test"
