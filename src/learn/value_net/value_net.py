import flax.linen as nn


class ValueNet(nn.Module):
    @nn.compact
    def __call__(self, board, sum_dice):

        board = nn.relu(nn.Conv(4, (1, 3))(board))
        board = nn.relu(nn.Conv(1, (1, 3))(board))

        board = board.reshape(board.shape[0], -1)
        board = nn.relu(nn.Dense(32)(board))
        sum_dice = nn.relu(nn.Dense(32)(sum_dice))
        x = board + sum_dice
        x = nn.Dense(1)(x)
        return nn.sigmoid(x)
