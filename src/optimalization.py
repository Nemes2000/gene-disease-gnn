import wandb
from config import Config
from train import train_node_classifier


def optimalization(dataset):
    sweep_config = {
        'method': 'random'
    }

    parameters_dict = {
        'optimizer': {
            'values': ['adam', 'sgd', 'adamW']
        },
        "num_layers": {
            "values": [10,5,2]
        },
        'hidden_channels': {
            'values': [30, 20, 16]
        },
        "dropout_rate": {
            "values" : [0.5, 0.3, 0.1]
        }
    }

    parameters_dict.update({
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
    }
    )

    sweep_config['parameters'] = parameters_dict
    if Config.test_dataset:
        project_name = "gnn_test_logs"
    else:
        project_name = "gnn_logs"

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    wandb.agent(sweep_id=sweep_id, function=wrapped_opt_train_function(dataset=dataset), count=Config.optimalization_step)

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

        train_node_classifier(dataset=dataset)
        