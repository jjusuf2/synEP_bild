import numpy as np

def _norm_ppf(p):
    """
    Inverse CDF (quantile) of the standard normal distribution.

    Acklam's rational approximation (|error| < 1.15e-9), used so this module
    has no SciPy dependency. For p=0.975 this returns 1.959964 (the 95% CI z).
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

def get_durations(profile):
    durations = []

    profile = np.asarray(profile)
    n = len(profile)
    
    i = 0
    while i < n:
        if profile[i] == 1:
            start = i
            while i < n and profile[i] == 1:
                i += 1
            end = i  # exclusive
            
            lifetime = end - start
            
            # censored if touching either boundary
            censored = (start == 0) or (end == n)
            
            durations.append(lifetime)
        else:
            i += 1
    
    return durations

def extract_lifetimes(tracks):
    """
    tracks: list of 1D lists/arrays of 0/1
    
    Returns:
        durations: list of lifetimes
        events: list of booleans (True = observed, False = censored)
    """
    durations = []
    events = []
    
    for track in tracks:
        track = np.asarray(track)
        n = len(track)
        
        i = 0
        while i < n:
            if track[i] == 1:
                start = i
                while i < n and track[i] == 1:
                    i += 1
                end = i  # exclusive
                
                lifetime = end - start
                
                # censored if touching either boundary
                censored = (start == 0) or (end == n)
                
                durations.append(lifetime)
                events.append(not censored)
            else:
                i += 1
    
    return np.array(durations), np.array(events)

def kaplan_meier(durations, events, alpha=0.05):
    """
    Compute Kaplan-Meier survival curve with pointwise confidence intervals.

    Confidence intervals use Greenwood's formula for the variance together with
    the log-log (complementary log-log) transformation, so the bounds always lie
    within [0, 1]. This matches the default in R's `survival` package and
    Python's `lifelines`.

    Parameters:
        durations: array of observed lifetimes
        events: boolean array (True = observed event, False = censored)
        alpha: significance level; alpha=0.05 -> 95% CI

    Returns:
        times: unique event times
        survival: survival probability at each time
        ci_lower: lower bound of the (1-alpha) CI at each time
        ci_upper: upper bound of the (1-alpha) CI at each time
    """
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]

    unique_times = np.unique(durations)

    n = len(durations)
    at_risk = n
    survival = 1.0

    # running sum for Greenwood's formula: sum_i d_i / (n_i * (n_i - d_i))
    greenwood_sum = 0.0

    times = []
    surv_probs = []
    var_terms = []

    for t in unique_times:
        mask = durations == t
        d_i = np.sum(events[mask])      # observed events
        c_i = np.sum(~events[mask])     # censored

        if at_risk > 0:
            if d_i > 0:
                survival *= (1 - d_i / at_risk)
                # Greenwood increment; guard against the n_i == d_i case
                denom = at_risk * (at_risk - d_i)
                if denom > 0:
                    greenwood_sum += d_i / denom
                times.append(t)
                surv_probs.append(survival)
                var_terms.append(greenwood_sum)

        at_risk -= (d_i + c_i)

    # Prepend the initial point at t=0, where everyone is at risk and S(0) = 1.
    times.insert(0, 0.0)
    surv_probs.insert(0, 1.0)
    var_terms.insert(0, 0.0)

    times = np.array(times)
    surv_probs = np.array(surv_probs)
    var_terms = np.array(var_terms)

    z = _norm_ppf(1 - alpha / 2)

    ci_lower = np.full_like(surv_probs, np.nan, dtype=float)
    ci_upper = np.full_like(surv_probs, np.nan, dtype=float)

    # Log-log transformation: CI on log(-log S), back-transformed to S.
    # Valid where 0 < S < 1; at S == 1 the variance is degenerate (CI = [1, 1]).
    valid = (surv_probs > 0) & (surv_probs < 1)
    log_s = np.log(surv_probs[valid])
    # Var(log(-log S)) = (1 / (log S)^2) * greenwood_sum
    se_loglog = np.sqrt(var_terms[valid]) / np.abs(log_s)
    ci_lower[valid] = surv_probs[valid] ** np.exp(z * se_loglog)
    ci_upper[valid] = surv_probs[valid] ** np.exp(-z * se_loglog)

    # S == 1 (no events yet) -> CI collapses to 1
    ci_lower[surv_probs >= 1] = 1.0
    ci_upper[surv_probs >= 1] = 1.0
    # S == 0 -> CI collapses to 0
    ci_lower[surv_probs <= 0] = 0.0
    ci_upper[surv_probs <= 0] = 0.0

    return times, surv_probs, ci_lower, ci_upper

def median_survival(times, survival):
    """
    Median survival = first time S(t) <= 0.5
    """
    if len(survival) == 0:
        return np.nan
    
    below = np.where(survival <= 0.5)[0]
    if len(below) == 0:
        return np.inf  # never drops below 0.5
    
    return times[below[0]]

def km_median_lifetime(tracks, alpha=0.05, return_ci=False):
    durations, events = extract_lifetimes(tracks)
    times, survival, ci_lower, ci_upper = kaplan_meier(durations, events, alpha=alpha)
    median = median_survival(times, survival)
    if return_ci:
        return median, times, survival, ci_lower, ci_upper
    # Backward-compatible 3-tuple (median, times, survival) by default
    return median, times, survival

def km_median_bootstrap(tracks, n_boot=1000, alpha=0.05, seed=None):
    """
    Bootstrap standard deviation (and CI) of the KM median survival time.

    The KM median has no simple closed-form standard error, so we resample the
    tracks (the independent units) with replacement, recompute the KM median for
    each resample, and report the spread of those medians.

    Parameters:
        tracks: list/array of 0/1 tracks (same input as km_median_lifetime)
        n_boot: number of bootstrap resamples
        alpha: significance level; alpha=0.05 -> 95% percentile CI
        seed: optional int for reproducibility

    Returns:
        median: point estimate of the median on the full data
        std: standard deviation of the bootstrap medians (finite values only)
        ci: (lower, upper) percentile CI of the median
        boot_medians: array of all bootstrap medians (may contain inf for
                      resamples whose curve never reaches 0.5)
    """
    rng = np.random.default_rng(seed)
    tracks = list(tracks)
    n = len(tracks)

    median = km_median_lifetime(tracks)[0]

    boot_medians = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resample = [tracks[i] for i in idx]
        boot_medians[b] = km_median_lifetime(resample)[0]

    finite = boot_medians[np.isfinite(boot_medians)]
    std = np.std(finite, ddof=1) if finite.size > 1 else np.nan
    if finite.size > 0:
        ci = (np.percentile(finite, 100 * alpha / 2),
              np.percentile(finite, 100 * (1 - alpha / 2)))
    else:
        ci = (np.nan, np.nan)

    return median, std, ci, boot_medians
