def nll(dist, y):
    return -dist.log_prob(y)
