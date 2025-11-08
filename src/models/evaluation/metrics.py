import numpy as np
from scipy.stats import pearsonr, spearmanr


def mse(preds, targets):
    preds = preds.unsqueeze(0)
    targets = targets.unsqueeze(0)
    mse_ = ((preds - targets) ** 2).mean(axis=(1, 2))
    results = list(mse_)
    return results


def point_score(locus, radius, matrix, pseudocount):
    l_edge = max(locus - radius, 0)
    r_edge = min(locus + radius, len(matrix))
    l_mask = matrix[l_edge : locus, l_edge : locus]
    r_mask = matrix[locus : r_edge, locus : r_edge]
    center_mask = matrix[l_edge : locus, locus : r_edge]
    score = (max(l_mask.mean(), r_mask.mean()) +  pseudocount) /\
            (center_mask.mean() + pseudocount)
    return score


def chr_score(matrix, res=5000, radius=125000, pseudocount_coeff=30, pseudocount=None):
    pseudocount = matrix.mean() * pseudocount_coeff
    pixel_radius = int(radius / res)
    scores = []
    for loc_i, loc in enumerate(range(len(matrix))):
        scores.append(point_score(loc, pixel_radius, matrix, pseudocount))
    return scores


def insulation_corr(preds, targets):
    scores_pearson = []
    scores_spearman = []
    preds = preds.unsqueeze(0)
    targets= targets.unsqueeze(0)
    for pred, target in zip(preds, targets):
        c = float(target.mean()) * 30
        pred_insu = np.array(chr_score(pred, pseudocount=c))
        label_insu = np.array(chr_score(target, pseudocount=c))
        nas = np.logical_or(np.isnan(pred_insu), np.isnan(label_insu))
        if nas.sum() == len(pred):
            scores_pearson.append(np.nan)
            scores_spearman.append(np.nan)
        else:
            metric_pearson, p_val = pearsonr(pred_insu[~nas], label_insu[~nas])
            scores_pearson.append(metric_pearson)
            metric_spearman, p_val = spearmanr(pred_insu[~nas], label_insu[~nas])
            scores_spearman.append(metric_spearman)
    return scores_pearson, scores_spearman
