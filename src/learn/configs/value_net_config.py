import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""

    config = ml_collections.ConfigDict()
    config.batch_size = 128
    config.path_dataset = "data/games/games_resumes.pickle"
    config.models_dir = "src/learn/models/value_net/"

    config.learning_rate = 1e-2
    config.momentum = 0.9

    config.num_epochs = 20
    return config
