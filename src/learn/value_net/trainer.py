from multiprocessing import dummy
import jax.numpy as jnp

from src.data.datasets.actions_dataloader_generator import ActionsDatasetGenerator
from src.data.datasets.embedder import Embedder
from .value_net import ValueNet

import ml_collections
from torch.utils import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax


@jax.jit
def apply_model(state, board, sum_dice, score):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        pred_score = state.apply_fn({'params': params}, board, sum_dice)
        loss = jnp.mean(jnp.square(pred_score - score))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return grads, loss


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds):
    """Train for a single epoch."""

    epoch_loss = []
    for batch in train_ds:
        board, sum_dice, score = batch
        grads, loss = apply_model(state, board, sum_dice, score)
        state = update_model(state, grads)
        epoch_loss += [loss]
    train_loss = np.mean(epoch_loss)
    return state, train_loss


def test_epoch(state, test_ds):
    """Train for a single epoch."""

    test_loss = []

    for batch in test_ds:
        board, sum_dice, score = batch
        _, loss = apply_model(state, board, sum_dice, score)
        test_loss += [loss]
    test_loss = np.mean(test_loss)
    return test_loss


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    value_net = ValueNet()

    dummy_board = jnp.ones((1, 9, 2))
    dummy_sum_dice = jnp.ones((1, 12))

    params = value_net.init(rng, dummy_board, dummy_sum_dice)['params']
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=value_net.apply,
                                         params=params,
                                         tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
    """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    The train state (which includes the `.params`).
  """

    adg = ActionsDatasetGenerator(config.path_dataset, config.batch_size,
                                  Embedder())
    train_ds, val_ds, test_ds = adg.split()

    rng = jax.random.PRNGKey(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.add_hparams(dict(config), dict())

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        state, train_loss = train_epoch(state, train_ds)
        test_loss = test_epoch(state, test_ds)

        print('epoch:% 3d, train_loss: %.4f, test_loss: %.4f' %
              (epoch, train_loss, test_loss))

        summary_writer.add_scalars(
            'loss',
            {
                'train': train_loss,
                'test': test_loss,
            },
            global_step=epoch,
        )

    summary_writer.flush()
    return state
