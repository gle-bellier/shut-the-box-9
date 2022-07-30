import numpy as np


class Embedder:

    def __init__(self) -> None:
        pass

    def __one_hot(self, i: int, nb_class: int) -> np.ndarray:
        """Compute one hot embedding.

        Args:
            i (int): index of the class.
            nb_class (int): number of classes.

        Returns:
            np.ndarray: one hot embedding
        """

        assert i < nb_class, f"Index {i} greater than (nb class {nb_class})-1"
        assert i > 0, "Index must be greater than 0."
        return np.eye(nb_class)[i]

    def __score_embedding(self, score: int) -> float:
        """Compute the score embedding i.e. score
        divided by the worst scenario score (=45),

        Args:
            score (int): score.

        Returns:
            float: normalized score.
        """
        return score / 45

    def embed(self, x: dict) -> dict:
        board = x["board"]
        sum_dice = self.__one_hot(x["sum_dice"], 12)
        score = self.__score_embedding(x["score"])

        return board, sum_dice, score
