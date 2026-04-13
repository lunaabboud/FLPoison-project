import numpy as np
from fl.client import Client
from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry


@attacker_registry
@actor('attacker', 'model_poisoning', 'omniscient')
class HIDRA(MPBase, Client):
    """
    [Attacking Byzantine Robust Aggregation in High Dimensions](https://ieeexplore.ieee.org/document/10646844) - IEEE S&P '24
    """

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        
        """
        knowledge (str): partial (only know malicious ones), full (know all)
        predicted_fraction_adv (float): the fraction of malicious clients assumed by the attacker. Full knowledge attacker knows it, while partial knowledge attacker does not just randomly predict it.
        """
        self.default_attack_params = {
            'knowledge': "partial",
            "predicted_fraction_adv": 0.2,
            "threshold": 1e-5
            }
        self.update_and_set_attr()


    def omniscient(self, clients):
        eps = self.predicted_fraction_adv
        malicious_indices = np.array([i.worker_id for i in clients if i.category == "attacker"], dtype=int)
        updates = np.array([i.update for i in clients])
        
        max_eigenvector = power_iteration(updates)
        max_variance = compute_variance(updates, max_eigenvector)
        threshold = 1.1 * max_variance
                
        benign_mean = np.mean(
                    updates[malicious_indices] if self.knowledge == "partial" else updates,
                    axis=0)
        
        if np.linalg.norm(benign_mean) == 0:
            return updates[:self.args.num_adv]
        
        # normalization to a unit vector serving as direction
        unit_vec = benign_mean / np.linalg.norm(benign_mean)

        # calculate the magnitude of the malicious deviation
        variance_diff = (np.sqrt(20) - 1) * threshold
        self.args.logger.info(f"Variance difference: {variance_diff}, Threshold: {threshold}")
        deviation = np.sqrt(variance_diff/(eps**2 + (1-eps)**2*eps))
        attack_vec = benign_mean - unit_vec*deviation
        return np.tile(attack_vec, (self.args.num_adv, 1))

def power_iteration(grads):
    rand_vector = np.random.randn(grads.shape[1])
    rand_vector = rand_vector / np.linalg.norm(rand_vector)
    
    num_iterations = 100
    current_eigenvector = rand_vector
    mean = np.mean(grads, axis=0)
    for itr in range(num_iterations):
        next_eigenvector = np.zeros(current_eigenvector.shape)
        for i in range(grads.shape[0]):
            next_eigenvector += np.dot(grads[i] - mean, current_eigenvector) * (grads[i] - mean)
        
        next_eigenvector = next_eigenvector/grads.shape[0]
        next_eigenvector = next_eigenvector/np.linalg.norm(next_eigenvector)
        current_eigenvector = next_eigenvector
        
    # print(f"Start Variance: {compute_variance(grads, rand_vector)}   End Variance: {compute_variance(grads, current_eigenvector)}")
    return current_eigenvector

def compute_variance(grads, proj_vector):
    projections = [np.dot(grad, proj_vector) for grad in grads]
    variance = np.var(projections)
    return variance