"""Best hyperparameters found in the seminar report (tuned on the validation users)."""

BEST_PARAMS = {
    'popularity': {'ml-20m': {}, 'netflix': {}, 'msd': {}},
    'itemknn': {
        'ml-20m': dict(similarity='cosine', num_neighbors=161, alpha=1.0),
        'netflix': dict(similarity='pearson', num_neighbors=22, alpha=1.0,
                        enable_average_bias=True, l1_normalization=True),
        'msd': dict(similarity='pearson', num_neighbors=10, alpha=1.0,
                    enable_average_bias=True, l1_normalization=True),
    },
    'ease': {
        'ml-20m': dict(lmbda=500),
        'netflix': dict(lmbda=1000),
        'msd': dict(lmbda=200),
    },
    'edlae': {
        'ml-20m': dict(lmbda=300, p=0.33),
        'netflix': dict(lmbda=500, p=0.33),
        'msd': dict(lmbda=70, p=0.25),
    },
    'dlae': {
        'ml-20m': dict(lmbda=300, p=0.33),
        'netflix': dict(lmbda=500, p=0.33),
        'msd': dict(lmbda=70, p=0.25),
    },
    'mrf_dense': {
        'ml-20m': dict(lmbda=619, mean_removal=True),
        'netflix': dict(lmbda=1195, mean_removal=True),
        'msd': dict(lmbda=130, mean_removal=True),
    },
    'recvae': {
        'ml-20m': dict(gamma=0.005, n_epochs=50),
        'netflix': dict(gamma=0.0035, n_epochs=50),
        'msd': dict(gamma=0.01, n_epochs=100),
    },
    'fm': {
        'ml-20m': dict(velocity_param='denoiser-gated', keep_prob=0.5, weight_decay=0.1, epochs=30),
    },
    # sparse models: settings for a density of ~0.5% of the item-item matrix
    'mrf': {
        'ml-20m': dict(lmbda=4.0, alpha=0.75, threshold=0.452, r=0.0, max_in_col=1000),
        'netflix': dict(lmbda=5.0, alpha=0.75, threshold=1.045, r=0.0, max_in_col=1000),
        'msd': dict(lmbda=2.0, alpha=0.75, threshold=0.1093, r=0.0, max_in_col=1000),
    },
    'admm': {
        'ml-20m': dict(lambda1=9.4, lambda2=10, p=0.33, rho=200, n_iter=100),
        'netflix': dict(lambda1=27.80, lambda2=0, p=0.33, rho=200, n_iter=100),
        'msd': dict(lambda1=4.96, lambda2=10, p=0.25, rho=200, n_iter=100),
    },
}

# MRFApprox settings from the report (Table A.2), per target density (%) and r
MRF_SWEEP = {
    'ml-20m': [
        dict(density=0.1, alpha=0.75, threshold=0.882, r=0.0, lmbda=4.0),
        dict(density=0.1, alpha=0.75, threshold=0.876, r=0.1, lmbda=3.0),
        dict(density=0.1, alpha=0.75, threshold=0.888, r=0.5, lmbda=2.0),
        dict(density=0.5, alpha=0.75, threshold=0.452, r=0.0, lmbda=4.0),
        dict(density=0.5, alpha=0.50, threshold=2.253, r=0.1, lmbda=18.0),
        dict(density=0.5, alpha=0.50, threshold=2.252, r=0.5, lmbda=14.0),
    ],
    'netflix': [
        dict(density=0.1, alpha=0.75, threshold=2.047, r=0.0, lmbda=6.0),
        dict(density=0.1, alpha=0.75, threshold=2.033, r=0.1, lmbda=4.0),
        dict(density=0.1, alpha=0.75, threshold=2.034, r=0.5, lmbda=2.0),
        dict(density=0.5, alpha=0.75, threshold=1.045, r=0.0, lmbda=5.0),
        dict(density=0.5, alpha=0.75, threshold=1.035, r=0.1, lmbda=5.0),
        dict(density=0.5, alpha=0.75, threshold=1.036, r=0.5, lmbda=3.0),
    ],
    'msd': [
        dict(density=0.1, alpha=0.75, threshold=0.370, r=0.0, lmbda=2.0),
        dict(density=0.1, alpha=0.50, threshold=2.006, r=0.1, lmbda=6.0),
        dict(density=0.1, alpha=0.75, threshold=0.363, r=0.5, lmbda=2.0),
        dict(density=0.5, alpha=0.75, threshold=0.1093, r=0.0, lmbda=2.0),
        dict(density=0.5, alpha=0.75, threshold=0.1012, r=0.1, lmbda=2.0),
        dict(density=0.5, alpha=0.50, threshold=0.539, r=0.5, lmbda=3.0),
    ],
}

# Sparsifying a fitted EDLAE: magnitude thresholds |B_ij| > t, per target density (%)
MAGNITUDE_THRESHOLDS = {
    'ml-20m': {0.1: 0.0122, 0.5: 0.00704, 1.0: 0.00522, 2.0: 0.00366, 5.0: 0.00208},
}

# Sparsifying a fitted EDLAE with the pattern |C_alpha| > threshold: density (%) -> (threshold, alpha)
CORRELATION_THRESHOLDS = {
    'ml-20m': {0.1: (6.580, 0.5), 0.5: (2.64, 0.5), 1.0: (9.23, 0.25), 2.0: (4.73, 0.25), 5.0: (1.69, 0.25)},
    'netflix': {0.1: (2.047, 0.75), 0.5: (1.096, 0.75), 1.0: (0.772, 0.75), 2.0: (0.509, 0.75), 5.0: (1.764, 0.5)},
    'msd': {0.1: (0.37, 0.75), 0.5: (0.609, 0.5), 1.0: (0.399, 0.5), 2.0: (0.0511, 0.75), 5.0: (0.00644, 1.0)},
}
