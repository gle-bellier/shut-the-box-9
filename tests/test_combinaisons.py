import pytest
from src.game.rules import combinaisons


def test_combinaisons():

    for i in range(3, 13):
        # testing each dice sum

        actions = combinaisons[i]
        print(actions)

        for action in actions:
            # we add one to the index to have the tile value
            assert sum(
                [index + 1 for index in action]
            ) == i, f"Error. Combinaison {action} not summing to dice sum = {i}"
