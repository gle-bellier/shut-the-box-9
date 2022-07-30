from src.learn.configs.value_net_config import get_config
from src.learn.value_net.trainer import train_and_evaluate

if __name__ == "__main__":

    config = get_config()
    state = train_and_evaluate(config, workdir="tensorboard/")
