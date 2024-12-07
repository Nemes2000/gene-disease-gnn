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

    parameters_dict = {
        'optimizer': {
            'values': ['adam', 'adamW']
        }
    }

    parameters_dict.update({
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
        "num_layers" : {
            'distribution': 'uniform',
            'min': 1,
            'max': 10
        },
        "hidden_channels" : {
            'distribution': 'uniform',
            'min': 16,
            'max': 100
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
        project_name = "gnn_logs"

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

        Config.optimizer = Config.optimizer_map[config.optimizer]
        Config.num_layers = config.num_layers
        Config.hidden_channels = config.hidden_channels
        Config.dropout_rate = config.dropout_rate
        Config.learning_rate = config.learning_rate
        Config.weight_decay = config.weight_decay

        train_node_classifier(dataset=dataset)
        