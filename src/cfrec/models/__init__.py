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
    from .flow import FlowMatchingCF
    from .recvae import RecVAE
    MODELS['recvae'] = RecVAE
    MODELS['flow'] = FlowMatchingCF
except ImportError:
    pass

try:  # optional dependency: fmbayes (JAX), see environment-fm.yml
    from .fm import FMRecommender
    MODELS['fm'] = FMRecommender
except ImportError:
    pass
