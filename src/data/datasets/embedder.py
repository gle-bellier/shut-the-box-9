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
            np.ndarray: one hot embedding of shape (nb_class, ).
        """

        assert i - 1 < nb_class, f"Index {i} greater than (nb class {nb_class})-1"
        assert i > 0, "Index must be greater than 0."
        return np.eye(nb_class)[i - 1]

    def __score_embedding(self, score: int) -> np.ndarray:
        """Compute the score embedding i.e. score
        divided by the worst scenario score (=45),

        Args:
            score (int): score.

        Returns:
            np.ndarray: normalized score.
        """
        return np.array([score / 45])

    def embed(self, x: dict) -> dict:
        """Compute the embeddings.

        Args:
            x (dict): board, sum di(c)e, score dict.

        Returns:
            dict: board of shape (9, 2), sum di(c)e of 
            shape (12,), score (float).
        """
        board = x["board"]
        sum_dice = self.__one_hot(x["sum_dice"], 12)
        score = self.__score_embedding(x["score"])

        return board, sum_dice, score
