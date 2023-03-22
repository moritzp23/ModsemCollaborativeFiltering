from numpy import log2, minimum, log

class Recall(object):
    """Recall metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        recall = len(hit_items) / minimum(self.topk, len(true_items))
        return recall


class DCG(object):
    """ Calculate discounted cumulative gain
    """
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        dcg = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                dcg += 1 / log2(2 + i)
        return dcg

    
class NDCG(object):
    """Normalized discounted cumulative gain metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        dcg_fn = DCG(k=self.topk)
        m = minimum(self.topk, len(true_items))
        idcg = dcg_fn(true_items[:m], true_items)
        dcg = dcg_fn(topk_items, true_items)
        return dcg / (idcg + 1e-12)
    
    
class NDCG_old(object):
    """Normalized discounted cumulative gain metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        dcg_fn = DCG(k=self.topk)
        idcg = dcg_fn(true_items[:self.topk], true_items)
        dcg = dcg_fn(topk_items, true_items)
        return dcg / (idcg + 1e-12)


class Recall_BARS(object):
    """Recall metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        hit_items = set(true_items) & set(topk_items)
        recall = len(hit_items) / (len(true_items) + 1e-12)
        return recall
    
    
class DCG_BARS(object):
    """ Calculate discounted cumulative gain
    """
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        true_items = set(true_items)
        dcg = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                dcg += 1 / log(2 + i)
        return dcg


class NDCG_BARS(object):
    """Normalized discounted cumulative gain metric."""
    def __init__(self, k=1):
        self.topk = k

    def __call__(self, topk_items, true_items):
        topk_items = topk_items[:self.topk]
        dcg_fn = DCG_BARS(k=self.topk)
        idcg = dcg_fn(true_items[:self.topk], true_items)
        dcg = dcg_fn(topk_items, true_items)
        return dcg / (idcg + 1e-12)