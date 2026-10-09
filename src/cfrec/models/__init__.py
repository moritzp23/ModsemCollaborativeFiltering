from .admm import ADMMSlim
from .base import ItemItemModel, Recommender
from .itemknn import ItemKNN
from .linear import DLAE, EASE, EDLAE, MRFDense
from .lowrank import LowRank, LowRankFactorization
from .mrf import MRFApprox
from .popularity import MostPopular

MODELS = {
    'popularity': MostPopular,
    'itemknn': ItemKNN,
    'ease': EASE,
    'edlae': EDLAE,
    'dlae': DLAE,
    'mrf_dense': MRFDense,
    'mrf': MRFApprox,
    'admm': ADMMSlim,
}

try:  # optional dependency: pip install -e ".[vae]"
    from .recvae import RecVAE
    MODELS['recvae'] = RecVAE
except ImportError:
    pass
