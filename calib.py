import json
import math
import os


def apply_temp(dist, T):
    """Apply temperature scaling to a probability distribution."""
    if not dist:
        return {}
    if T == 1.0:
        return dict(dist)
    
    # p^(1/T) in log space with the max subtracted (no underflow at small T), zeros stay zeros
    logs = {k: math.log(p) / T for k, p in dist.items() if p > 0.0}
    m = max(logs.values(), default=0.0)
    scaled = {k: math.exp(logs[k] - m) if k in logs else 0.0 for k in dist}
    total = sum(scaled.values())
    if total == 0.0:
        return {k: 0.0 for k in dist}
    
    return {k: v / total for k, v in scaled.items()}


def nll(samples, T):
    """Negative log-likelihood of samples under temperature T."""
    if not samples:
        return 0.0
    
    total = 0.0
    for dist, true_option in samples:
        calibrated = apply_temp(dist, T)
        p_true = calibrated.get(true_option, 0.0)
        p_true = max(p_true, 1e-12)
        total += -math.log(p_true)
    
    return total / len(samples)


def fit_temp(samples, lo=0.05, hi=10.0):
    """Find the temperature T in [lo, hi] that minimizes NLL."""
    if not samples:
        return 1.0
    
    # Log-spaced grid of 200 points
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    n_grid = 200
    grid = [math.exp(log_lo + (log_hi - log_lo) * i / (n_grid - 1)) for i in range(n_grid)]
    
    # Evaluate NLL at each grid point
    best_idx = 0
    best_val = nll(samples, grid[0])
    for i in range(1, n_grid):
        val = nll(samples, grid[i])
        if val < best_val:
            best_val = val
            best_idx = i
    
    # Golden-section refine around the best grid point
    # Define search interval around best_idx
    left_idx = max(0, best_idx - 1)
    right_idx = min(n_grid - 1, best_idx + 1)
    
    a = grid[left_idx]
    b = grid[right_idx]
    
    # Golden-section search
    gr = (math.sqrt(5) - 1) / 2  # golden ratio conjugate
    
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    fc = nll(samples, c)
    fd = nll(samples, d)
    
    for _ in range(50):  # sufficient iterations for convergence
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = nll(samples, c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = nll(samples, d)
        
        if abs(b - a) < 1e-10:
            break
    
    # Return the best point found
    best_T = (a + b) / 2.0
    best_T = max(lo, min(hi, best_T))
    
    return best_T


def top(dist):
    """Return (option, p) of the highest probability, ties to first key."""
    if not dist:
        return (None, 0.0)
    
    best_key = None
    best_p = -1.0
    for k, p in dist.items():
        if p > best_p:
            best_p = p
            best_key = k
    
    return (best_key, best_p)


def brier(samples, T=1.0):
    """Brier score: mean over samples of sum over options of (p - y)^2."""
    if not samples:
        return 0.0
    
    total = 0.0
    for dist, true_option in samples:
        calibrated = apply_temp(dist, T)
        # Collect all options: dist keys plus the true option
        all_options = set(dist.keys())
        all_options.add(true_option)
        
        sample_score = 0.0
        for opt in all_options:
            p = calibrated.get(opt, 0.0)
            y = 1.0 if opt == true_option else 0.0
            sample_score += (p - y) ** 2
        
        total += sample_score
    
    return total / len(samples)


def ece(samples, T=1.0, bins=10):
    """Expected calibration error on top confidence."""
    if not samples:
        return 0.0
    
    # Compute calibrated top confidences and accuracies
    confs = []
    accs = []
    for dist, true_option in samples:
        calibrated = apply_temp(dist, T)
        top_opt, top_p = top(calibrated)
        confs.append(top_p)
        accs.append(1.0 if top_opt == true_option else 0.0)
    
    N = len(samples)
    
    # Equal-width bins over [0, 1]
    bin_size = 1.0 / bins
    bin_counts = [0] * bins
    bin_conf_sum = [0.0] * bins
    bin_acc_sum = [0.0] * bins
    
    for conf, acc in zip(confs, accs):
        # conf 1.0 goes in the last bin
        if conf >= 1.0:
            bin_idx = bins - 1
        else:
            bin_idx = int(conf / bin_size)
            if bin_idx >= bins:
                bin_idx = bins - 1
        
        bin_counts[bin_idx] += 1
        bin_conf_sum[bin_idx] += conf
        bin_acc_sum[bin_idx] += acc
    
    total_ece = 0.0
    for i in range(bins):
        if bin_counts[i] > 0:
            mean_conf = bin_conf_sum[i] / bin_counts[i]
            mean_acc = bin_acc_sum[i] / bin_counts[i]
            total_ece += (bin_counts[i] / N) * abs(mean_acc - mean_conf)
    
    return total_ece


def report(samples, T):
    """Return a report dict with n, acc, mean_conf, brier, ece."""
    if not samples:
        return {'n': 0, 'acc': 0.0, 'mean_conf': 0.0, 'brier': 0.0, 'ece': 0.0}
    
    n = len(samples)
    correct = 0
    conf_sum = 0.0
    
    for dist, true_option in samples:
        calibrated = apply_temp(dist, T)
        top_opt, top_p = top(calibrated)
        conf_sum += top_p
        if top_opt == true_option:
            correct += 1
    
    acc = correct / n
    mean_conf = conf_sum / n
    
    return {
        'n': n,
        'acc': acc,
        'mean_conf': mean_conf,
        'brier': brier(samples, T),
        'ece': ece(samples, T),
    }


def route(conf, auto=0.9, ask=0.6):
    """Route based on confidence: 'auto', 'ask', or 'escalate'."""
    if conf is None:
        return 'escalate'
    if conf >= auto:
        return 'auto'
    if conf >= ask:
        return 'ask'
    return 'escalate'


def auto_threshold(samples, T, target_acc=0.95):
    """Find the smallest threshold t such that samples with conf >= t have accuracy >= target_acc."""
    if not samples:
        return None
    
    # Compute calibrated top confidences and accuracies
    data = []
    for dist, true_option in samples:
        calibrated = apply_temp(dist, T)
        top_opt, top_p = top(calibrated)
        acc = 1.0 if top_opt == true_option else 0.0
        data.append((top_p, acc))
    
    # Get distinct confidences as candidates
    distinct_confs = sorted(set(c for c, _ in data))
    
    # Try each candidate threshold from smallest to largest
    for t in distinct_confs:
        # Samples with conf >= t
        selected = [(c, a) for c, a in data if c >= t]
        if not selected:
            continue
        acc = sum(a for _, a in selected) / len(selected)
        if acc >= target_acc:
            return t
    
    return None


def seq_dist(token_logprobs, norm='mean'):
    """Convert token log-probs to a probability distribution via softmax."""
    if not token_logprobs:
        return {}
    
    if norm not in ('mean', 'sum'):
        raise ValueError(f"Unknown norm: {norm}")
    
    # Compute scores
    scores = {}
    for option, logprobs in token_logprobs.items():
        if not logprobs:
            scores[option] = None  # empty list means probability 0.0
        else:
            if norm == 'mean':
                scores[option] = sum(logprobs) / len(logprobs)
            else:  # norm == 'sum'
                scores[option] = sum(logprobs)
    
    # Separate options with scores from those without
    scored_options = [(k, v) for k, v in scores.items() if v is not None]
    zero_options = [k for k, v in scores.items() if v is None]
    
    if not scored_options:
        return {k: 0.0 for k in token_logprobs}
    
    # Softmax with max subtraction for numerical stability
    max_score = max(v for _, v in scored_options)
    exp_scores = {}
    for k, v in scored_options:
        exp_scores[k] = math.exp(v - max_score)
    
    total = sum(exp_scores.values())
    
    result = {}
    for k in token_logprobs:
        if k in exp_scores:
            result[k] = exp_scores[k] / total
        else:
            result[k] = 0.0
    
    return result


def save(path, d):
    """Write a dict as JSON to a file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f)


def load(path):
    """Read a JSON file, returning defaults if missing."""
    d = {'T': 1.0, 'auto': 0.9, 'ask': 0.6}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            d.update(json.load(f))                  # a partial file keeps the defaults for missing keys
    return d
