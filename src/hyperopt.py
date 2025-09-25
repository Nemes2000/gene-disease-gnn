import wandb
from config import Config
from train import train_node_classifier


def optimalization(dataset):
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        }
    }

    if Config.pr_disease_idx:
        parameters_dict = {}

        parameters_dict.update({
            "v_emb_dim" : {
                'distribution': 'q_log_uniform_values',
                'q': 1,
                'min': 1,
                'max': 100
            },
            "mt_hidden_1" : {
                'distribution': 'q_log_uniform_values',
                'q': 1,
                'min': 1,
                'max': 100
            },
            "mt_hidden_2" : {
                'distribution': 'q_log_uniform_values',
                'q': 1,
                'min': 16,
                'max': 100
            },
            'mt_lr': {
                'distribution': 'uniform',
                'min': 0,
                'max': 0.1
            },
            'mt_wd': {
                'distribution': 'uniform',
                'min': 0.0001,
                'max': 0.01
            },
            'mt_eps': {
                'distribution': 'uniform',
                'min': 0.000001,
                'max': 0.0001
            }
        }
        )
    else:
        parameters_dict = {
            'optimizer': {
                'values': ['adam', 'adamW']
            }
        }

        parameters_dict.update({
            "num_layers" : {
                'distribution': 'q_log_uniform_values',
                'q': 1,
                'min': 1,
                'max': 100
            },
            "hidden_channels" : {
                'distribution': 'q_log_uniform_values',
                'q': 1,
                'min': 16,
                'max': 100
            },
            'learning_rate': {
                'distribution': 'uniform',
                'min': 0,
                'max': 0.1
            },
            'weight_decay': {
                'distribution': 'uniform',
                'min': 0.0001,
                'max': 0.01
            },
            "dropout_rate" : {
                'distribution': 'uniform',
                'min': 0.01,
                'max': 0.05
            }
        }
        )

    sweep_config['parameters'] = parameters_dict
    if Config.test_dataset:
        project_name = "gnn_test_logs"
    else:
        project_name = Config.wandb_project_name

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    wandb.agent(sweep_id=sweep_id, function=wrapped_opt_train_function(dataset=dataset), count=Config.optimalization_step)
    wandb.teardown()

def wrapped_opt_train_function(dataset):
    def train_wrapper(config=None):
        optimalization_train(config=config, dataset=dataset)
    return train_wrapper

def optimalization_train(config=None, dataset=None):
    with wandb.init(config=config):
        config = wandb.config
        
        if Config.pr_disease_idx:
            Config.v_emb_dim = config.v_emb_dim
            Config.mt_wd = config.mt_wd
            Config.mt_eps = config.mt_eps
            Config.mt_lr = config.mt_lr
            Config.mt_hidden_1 = config.mt_hidden_1
            Config.mt_hidden_2 = config.mt_hidden_2
        else:
            Config.optimizer = Config.optimizer_map[config.optimizer]
            Config.num_layers = config.num_layers
            Config.hidden_channels = config.hidden_channels
            Config.dropout_rate = config.dropout_rate
            Config.learning_rate = config.learning_rate
            Config.weight_decay = config.weight_decay

        train_node_classifier(dataset=dataset)
        