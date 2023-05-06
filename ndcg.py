import numpy as np

def dcg_at_k(r, k):
    """
    Calculate DCG@k (Discounted Cumulative Gain).

    Parameters:
        r (list): A list of relevance scores.
        k (int): The number of items to consider.

    Returns:
        float: The DCG@k value.
    """
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain).

    Parameters:
        r (list): A list of relevance scores.
        k (int): The number of items to consider.

    Returns:
        float: The NDCG@k value.
    """
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

