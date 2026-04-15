from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry
import numpy as np
from scipy.stats import skew
from scipy.stats import zscore, skew
from scipy.spatial.distance import cosine

@aggregator_registry
class FedSign(AggregatorBase):
    def __init__(self, args, **kwargs):
        """
        [FedSIGN: A sign-based federated learning framework with privacy and
        robustness guarantees](https://www.sciencedirect.com/science/article/abs/pii/S016740482300384X) - Computers & Security 2023
        """
        super().__init__(args)
        self.default_defense_params = {
            "threshold_k": 3.0,  # MAD z-score 阈值
            # m_potential_malicious: number of potentially malicious clients (paper uses m)
            # used to compute phi = m*(m-1)/2 for threshold selection
            "m_potential_malicious": self.args.num_adv,
            }
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        # PAD: pairwise attack density filter
        """
        PAD filter based on pairwise cosine similarity of sign vectors.

        Steps implemented:
        - compute pairwise cosine similarity matrix A between sign vectors
          and map to [0,1] via (1 + cos)/2
        - collect upper-triangular (i>j) similarities, sort descending,
          pick the phi-th largest as Q_phi where phi = m*(m-1)/2
        - for each client i compute P_i = sum_j chi(A_ij - Q_phi) (chi=1 if >0)
        - compute Pm = mean(P_i)
        - ASSUMPTION: clients with attack density P_i < Pm are considered honest
          (paper text seems to have a reversal; this implementation treats
           high attack density as suspect/malicious and keeps those below the mean)

        Returns: (benign_indices_list, metadata_dict)
        """
        sign_updates = np.sign(updates)
        n = sign_updates.shape[0]
        if n <= 1:
            return list(range(n)), {'Q_phi': None, 'Pi': np.array([0]) if n==1 else np.array([]), 'Pm': 0}

        # compute cosine similarity matrix (rows are vectors)
        try:
            sim = cosine_similarity(sign_updates)
        except Exception:
            # fallback to manual safe computation
            norms = np.linalg.norm(sign_updates, axis=1)
            norms[norms == 0] = 1e-12
            sim = np.dot(sign_updates, sign_updates.T) / np.outer(norms, norms)

        # map to [0,1]
        sim = (1.0 + sim) / 2.0

        # upper triangular values (i>j)
        triu_idx = np.triu_indices(n, k=1)
        upper = sim[triu_idx]
        if upper.size == 0:
            Q_phi = 1.0
        else:
            sorted_desc = np.sort(upper)[::-1]
            # compute phi
            phi = int(self.m_potential_malicious * (self.m_potential_malicious - 1) // 2)
            if phi < 1:
                phi = 1
            num_pairs = upper.size
            # cap phi to number of pairs
            idx = min(phi - 1, num_pairs - 1)
            Q_phi = float(sorted_desc[idx])

        # compute P_i = number of clients with similarity > Q_phi
        Pi = np.sum(sim > Q_phi, axis=1)
        Pm = float(np.mean(Pi))
 
        pad_benign_indices = np.where(Pi >= Pm)[0].tolist()
        pad_meta = {'Q_phi': Q_phi, 'Pi': Pi, 'Pm': Pm}
    
        self.args.logger.info(f"PAD benign indices: {pad_benign_indices} (Q_phi={pad_meta.get('Q_phi')}, Pm={pad_meta.get('Pm')})")

        self.args.logger.info(f"Malicious client indices: {set(range(len(updates))) - set(pad_benign_indices)}\n")

        return np.mean(updates[pad_benign_indices], axis=0)
