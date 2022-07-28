import numpy as np
from copy import deepcopy

from src.game.board import Board
from src.agents.agent import Agent


class Game:

    def __init__(self, board: Board, agent: Agent) -> None:
        """Initialize the game.

        Args:
            board (Board): initial board of the game.
            agent (Agent): agent playing the game.
        """

        self.board = board
        self.agent = agent

        # history of the played moves
        self.history = []

        # state = True while the game is on
        self.state = True

    def step(self, verbose=False) -> bool:
        """Compute one step of the game.
        Args:
            verbose (bool): if True then steps are printed.
            Defaults to False.

        Returns:
            bool: True if the game is still on at the
            end of this step, else False.
        """

        # 1 step: roll the dice
        if self.board.check_under_six():
            # only one die
            sum_dice = np.random.randint(1, 7)
        else:
            # roll two dice
            sum_dice = np.random.randint(1, 7) + np.random.randint(1, 7)

        # check if there is any legal action for this dice sum
        if self.board.get_legal_actions(sum_dice) == []:
            if verbose:
                print(f"No legal action. Sum of di(c)e = {sum_dice}")

            # add step to the history of the game
            step_record = {
                "sum_dice": sum_dice,
                "action": None,
                "board": deepcopy(self.board)
            }
            self.history += [step_record]
            return False

        # if there is then ask the agent to choose.
        # this action is legal (the agent tests this assumption).
        action = self.agent.select_action(self.board, sum_dice)
        self.board.play_action(action)

        if verbose:
            tiles = [index + 1 for index in action]
            print(f"Sum of dice={sum_dice}, agent plays -> {tiles}")
            print("Board after move:")
            print(self.board)

        # add step to the history of the game
        step_record = {
            "sum_dice": sum_dice,
            "action": action,
            "board": deepcopy(self.board)
        }
        self.history += [step_record]

        # check if the game is finished or not
        return not self.board.check_end()

    def run_game(self, verbose=False) -> float:
        """Run the game of shut the box.

        Args:
            verbose (bool, optional): if True then extra information are printed. 
            Defaults to False.

        Returns:
            float: score obtained by the agent.
        """

        while self.step(verbose):
            pass

        score = self.evaluate()

        # add score to the history
        self.history += [{"score": score}]

        if verbose:
            print(f"Score = {score}.")
        return score

    def evaluate(self) -> float:
        """Evaluate the score at the end of the game.
        Sum of the values of the tiles that are still up
        divided by the result in the worst case. (resign)

        Returns:
            float: score between 0 and 1
        """

        digits = np.arange(1, 10, 1)
        up_values = self.board.board[:, 0] * digits
        return np.sum(up_values) / np.sum(digits)
