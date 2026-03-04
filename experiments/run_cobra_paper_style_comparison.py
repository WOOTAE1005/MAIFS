#!/usr/bin/env python3
"""
COBRA paper-style protocol comparison with upgraded baselines:
1) stronger shared RM backbone (default: TF-IDF + SGD)
2) validation-driven COBRA hyperparameter search
3) robust COBRA aggregation variants (standard/trimmed/mom/huber)
4) DAAC pairwise ranking mode for HH-RLHF
5) learned COBRA+DAAC fusion gate

Datasets:
- SST-2 (train / validation(eval), because GLUE test labels are hidden)
- IMDB (train/test)
- Anthropic/hh-rlhf (train/test pairwise)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from scipy.stats import ttest_rel
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split

EPS = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TextRewardModel:
    vectorizer: Any
    clf: Any
    backbone: str

    def score(self, texts: Sequence[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        if self.backbone == "naive_bayes":
            x = x.maximum(0)
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(x)[:, 1]
        return sigmoid(self.clf.decision_function(x))


@dataclass(frozen=True)
class CobraConfig:
    algorithm: str
    robust_method: str
    trust_threshold: float = 0.7
    alpha: float = 0.3
    epsilon: float = 0.2
    beta: float = 2.0


def parse_int_list(s: str) -> List[int]:
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def parse_str_list(s: str) -> List[str]:
    return [v.strip() for v in s.split(",") if v.strip()]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def paired_pvalue(a: Sequence[float], b: Sequence[float]) -> float:
    stat = ttest_rel(a, b, nan_policy="omit")
    return float(stat.pvalue) if np.isfinite(stat.pvalue) else float("nan")


def summarize_runs(runs: List[float]) -> Dict[str, Any]:
    return {
        "mean": float(np.mean(runs)),
        "std": float(np.std(runs, ddof=1) if len(runs) > 1 else 0.0),
        "runs": runs,
    }


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def sample_rows(texts: List[str], labels: np.ndarray, max_samples: int, rng: np.random.Generator) -> Tuple[List[str], np.ndarray]:
    if max_samples <= 0 or len(texts) <= max_samples:
        return texts, labels
    idx = rng.choice(len(texts), size=max_samples, replace=False)
    idx.sort()
    return [texts[i] for i in idx], labels[idx]


def split_three_way(
    texts: List[str],
    labels: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=test_size + val_size, random_state=seed, stratify=labels
    )
    rel_val = val_size / (test_size + val_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=1.0 - rel_val, random_state=seed + 1, stratify=y_temp
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def train_text_reward_model(
    texts: Sequence[str],
    labels: np.ndarray,
    seed: int,
    backbone: str,
    tfidf_max_features: int,
) -> TextRewardModel:
    if backbone == "tfidf":
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=tfidf_max_features,
            lowercase=True,
            sublinear_tf=True,
        )
        clf = SGDClassifier(
            loss="log_loss",
            alpha=5e-6,
            max_iter=3000,
            tol=1e-4,
            random_state=seed,
        )
    elif backbone == "hashing":
        vectorizer = HashingVectorizer(
            n_features=2**18,
            alternate_sign=False,
            ngram_range=(1, 2),
            lowercase=True,
            norm="l2",
        )
        clf = SGDClassifier(
            loss="log_loss",
            alpha=5e-6,
            max_iter=3000,
            tol=1e-4,
            random_state=seed,
        )
    elif backbone == "char_ngram":
        # 형태론적 패턴 학습: 문자 수준 n-gram (word-boundary 포함)
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=50000,
            lowercase=True,
            sublinear_tf=True,
        )
        clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=3000,
            tol=1e-4,
            random_state=seed,
        )
    elif backbone == "naive_bayes":
        # 확률 모델 기반: 작은 어휘로 확률적 추론 (word TF-IDF + ComplementNB)
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            lowercase=True,
            sublinear_tf=False,
            min_df=1,
        )
        clf = ComplementNB(alpha=0.1)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    x = vectorizer.fit_transform(texts) if hasattr(vectorizer, "fit_transform") else vectorizer.transform(texts)
    # ComplementNB는 음수 입력을 허용하지 않으므로 clip
    if backbone == "naive_bayes":
        x = x.maximum(0)
    clf.fit(x, labels)
    return TextRewardModel(vectorizer=vectorizer, clf=clf, backbone=backbone)


def build_partition_models(
    texts: List[str],
    labels: np.ndarray,
    n_partitions: int,
    n_trusted: int,
    poison_flip_prob: float,
    trusted_score: float,
    untrusted_score: float,
    seed: int,
    backbone: str,
    tfidf_max_features: int,
) -> Tuple[List[TextRewardModel], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))
    chunks = np.array_split(idx, n_partitions)

    models: List[TextRewardModel] = []
    trusts = np.zeros(n_partitions, dtype=np.float64)
    is_trusted = np.zeros(n_partitions, dtype=np.int64)

    # 이종 에이전트 풀: backbone이 "diverse"이면 라운드로빈으로 3가지 백본을 할당한다.
    # Group A (1/3): word TF-IDF + SGD  (기존)
    # Group B (1/3): char n-gram + SGD  (형태론적 패턴)
    # Group C (1/3): small-vocab TF-IDF + ComplementNB  (확률 모델)
    diverse_backbones = ["tfidf", "char_ngram", "naive_bayes"]

    for i, chunk in enumerate(chunks):
        x_part = [texts[j] for j in chunk]
        y_part = labels[chunk].copy()
        trusted = i < n_trusted
        if not trusted:
            flips = rng.random(len(y_part)) < poison_flip_prob
            y_part[flips] = 1 - y_part[flips]
            trusts[i] = untrusted_score
        else:
            trusts[i] = trusted_score
            is_trusted[i] = 1

        agent_backbone = diverse_backbones[i % len(diverse_backbones)] if backbone == "diverse" else backbone
        model = train_text_reward_model(
            texts=x_part,
            labels=y_part,
            seed=seed + 101 * (i + 1),
            backbone=agent_backbone,
            tfidf_max_features=tfidf_max_features,
        )
        models.append(model)

    return models, trusts, is_trusted


def score_matrix(models: Sequence[TextRewardModel], texts: Sequence[str]) -> np.ndarray:
    return np.vstack([m.score(texts) for m in models])


def variance_profile_from_prob(prob_mat: np.ndarray) -> np.ndarray:
    return np.var(prob_mat, axis=1)


def _rot_weight_matrix(prob_mat: np.ndarray, trusts: np.ndarray, trust_threshold: float, alpha: float) -> np.ndarray:
    confs = np.abs(prob_mat - 0.5) * 2.0
    trusted_mask = (trusts >= trust_threshold).astype(np.float64)[:, None]
    base = trusts[:, None] * confs
    return base * (trusted_mask + (1.0 - trusted_mask) * alpha)


def _drwa_weight_matrix(prob_mat: np.ndarray, trusts: np.ndarray, variances: np.ndarray, epsilon: float) -> np.ndarray:
    confs = np.abs(prob_mat - 0.5) * 2.0
    max_var = float(np.max(variances) + EPS)
    var_factor = 1.0 - (variances / max_var)
    dyn_trust = np.clip(trusts + epsilon * var_factor, 0.0, 1.5)
    return dyn_trust[:, None] * confs


def _avga_weight_matrix(prob_mat: np.ndarray, trusts: np.ndarray, beta: float) -> np.ndarray:
    confs = np.abs(prob_mat - 0.5) * 2.0
    base = trusts[:, None] * confs
    conf_var = np.var(confs, axis=0, keepdims=True)
    temp = 1.0 + beta * conf_var
    scaled = base / np.maximum(temp, EPS)
    shifted = scaled - np.max(scaled, axis=0, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.maximum(np.sum(exps, axis=0, keepdims=True), EPS)


def _aggregate_signed(
    prob_mat: np.ndarray,
    weight_mat: np.ndarray,
    robust_method: str,
    trim_ratio: float,
    mom_groups: int,
    huber_c: float,
) -> np.ndarray:
    signed = weight_mat * (2.0 * prob_mat - 1.0)
    n_agents, n_samples = signed.shape

    if robust_method == "standard":
        return np.sum(signed, axis=0)

    scores = np.zeros(n_samples, dtype=np.float64)

    if robust_method == "trimmed":
        k = int(np.floor(n_agents * trim_ratio))
        if 2 * k >= n_agents:
            k = max(0, (n_agents - 1) // 2)
        for j in range(n_samples):
            vals = np.sort(signed[:, j])
            core = vals[k:n_agents - k] if k > 0 else vals
            if core.size == 0:
                core = vals
            scores[j] = float(np.sum(core))
        return scores

    if robust_method == "mom":
        g = int(np.clip(mom_groups, 2, n_agents))
        for j in range(n_samples):
            vals = signed[:, j]
            groups = np.array_split(vals, g)
            group_means = np.array([float(np.mean(grp)) for grp in groups], dtype=np.float64)
            scores[j] = float(np.median(group_means)) * n_agents
        return scores

    if robust_method == "huber":
        c = max(huber_c, 1e-3)
        for j in range(n_samples):
            vals = signed[:, j]
            center = float(np.median(vals))
            mad = float(np.median(np.abs(vals - center)) + EPS)
            r = (vals - center) / (c * mad)
            h = np.where(np.abs(r) <= 1.0, 1.0, 1.0 / (np.abs(r) + EPS))
            scores[j] = float(np.sum(vals * h))
        return scores

    raise ValueError(f"Unknown robust_method: {robust_method}")


def cobra_predict(
    prob_mat: np.ndarray,
    trusts: np.ndarray,
    variances: np.ndarray,
    cfg: CobraConfig,
    trim_ratio: float,
    mom_groups: int,
    huber_c: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cfg.algorithm == "rot":
        w = _rot_weight_matrix(prob_mat, trusts, trust_threshold=cfg.trust_threshold, alpha=cfg.alpha)
    elif cfg.algorithm == "drwa":
        w = _drwa_weight_matrix(prob_mat, trusts, variances=variances, epsilon=cfg.epsilon)
    elif cfg.algorithm == "avga":
        w = _avga_weight_matrix(prob_mat, trusts, beta=cfg.beta)
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

    score = _aggregate_signed(
        prob_mat=prob_mat,
        weight_mat=w,
        robust_method=cfg.robust_method,
        trim_ratio=trim_ratio,
        mom_groups=mom_groups,
        huber_c=huber_c,
    )
    pred = (score >= 0.0).astype(np.int64)
    denom = np.sum(np.abs(w), axis=0) + EPS
    conf = np.clip(np.abs(score) / denom, 0.0, 1.0)
    return pred, conf, score


def build_cobra_grid(
    robust_methods: Sequence[str],
    rot_thresholds: Sequence[float],
    rot_alphas: Sequence[float],
    drwa_epsilons: Sequence[float],
    avga_betas: Sequence[float],
) -> List[CobraConfig]:
    grid: List[CobraConfig] = []
    for robust in robust_methods:
        for t in rot_thresholds:
            for a in rot_alphas:
                grid.append(CobraConfig(algorithm="rot", robust_method=robust, trust_threshold=t, alpha=a))
        for e in drwa_epsilons:
            grid.append(CobraConfig(algorithm="drwa", robust_method=robust, epsilon=e))
        for b in avga_betas:
            grid.append(CobraConfig(algorithm="avga", robust_method=robust, beta=b))
    return grid


def select_best_cobra_config(
    val_prob: np.ndarray,
    val_labels: np.ndarray,
    trusts: np.ndarray,
    variances: np.ndarray,
    grid: Sequence[CobraConfig],
    trim_ratio: float,
    mom_groups: int,
    huber_c: float,
) -> Tuple[CobraConfig, float]:
    best_cfg = grid[0]
    best_acc = -1.0
    best_conf = -1.0
    for cfg in grid:
        pred, conf, _ = cobra_predict(
            prob_mat=val_prob,
            trusts=trusts,
            variances=variances,
            cfg=cfg,
            trim_ratio=trim_ratio,
            mom_groups=mom_groups,
            huber_c=huber_c,
        )
        acc = accuracy(val_labels, pred)
        mconf = float(np.mean(conf))
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and mconf > best_conf):
            best_cfg = cfg
            best_acc = acc
            best_conf = mconf
    return best_cfg, best_acc


def daac_features_from_prob(prob_mat: np.ndarray, trusts: np.ndarray, trust_threshold: float = 0.7) -> np.ndarray:
    mean = np.mean(prob_mat, axis=0)
    std = np.std(prob_mat, axis=0)
    mn = np.min(prob_mat, axis=0)
    mx = np.max(prob_mat, axis=0)

    trusted_mask = trusts >= trust_threshold
    trusted_mean = np.mean(prob_mat[trusted_mask], axis=0) if np.any(trusted_mask) else mean
    untrusted_mean = np.mean(prob_mat[~trusted_mask], axis=0) if np.any(~trusted_mask) else mean

    trust_weighted = np.average(prob_mat, axis=0, weights=trusts + EPS)
    maj = (mean >= 0.5).astype(np.float64)
    disagreement = np.mean((prob_mat >= 0.5) != maj[None, :], axis=0)

    feat_blocks = [
        prob_mat,
        mean[None, :],
        std[None, :],
        mn[None, :],
        mx[None, :],
        trusted_mean[None, :],
        untrusted_mean[None, :],
        trust_weighted[None, :],
        disagreement[None, :],
    ]
    return np.vstack(feat_blocks).T


def daac_features_from_pair_delta(delta_mat: np.ndarray, trusts: np.ndarray, trust_threshold: float = 0.7) -> np.ndarray:
    mean = np.mean(delta_mat, axis=0)
    std = np.std(delta_mat, axis=0)
    mn = np.min(delta_mat, axis=0)
    mx = np.max(delta_mat, axis=0)

    trusted_mask = trusts >= trust_threshold
    trusted_mean = np.mean(delta_mat[trusted_mask], axis=0) if np.any(trusted_mask) else mean
    untrusted_mean = np.mean(delta_mat[~trusted_mask], axis=0) if np.any(~trusted_mask) else mean

    trust_weighted = np.average(delta_mat, axis=0, weights=trusts + EPS)
    sign_disagreement = np.mean((delta_mat >= 0.0) != (mean[None, :] >= 0.0), axis=0)

    feat_blocks = [
        delta_mat,
        mean[None, :],
        std[None, :],
        mn[None, :],
        mx[None, :],
        trusted_mean[None, :],
        untrusted_mean[None, :],
        trust_weighted[None, :],
        sign_disagreement[None, :],
    ]
    return np.vstack(feat_blocks).T


def gate_features(
    prob_mat: np.ndarray,
    cobra_pred: np.ndarray,
    cobra_conf: np.ndarray,
    cobra_score: np.ndarray,
    daac_prob: np.ndarray,
) -> np.ndarray:
    mean = np.mean(prob_mat, axis=0)
    std = np.std(prob_mat, axis=0)
    mn = np.min(prob_mat, axis=0)
    mx = np.max(prob_mat, axis=0)
    maj = (mean >= 0.5).astype(np.float64)
    disagreement = np.mean((prob_mat >= 0.5) != maj[None, :], axis=0)

    daac_pred = (daac_prob >= 0.5).astype(np.float64)
    daac_conf = np.abs(daac_prob - 0.5) * 2.0
    cobra_score_n = np.tanh(cobra_score)

    cols = [
        mean,
        std,
        mn,
        mx,
        disagreement,
        cobra_pred.astype(np.float64),
        cobra_conf,
        cobra_score_n,
        daac_prob,
        daac_conf,
        np.abs(cobra_pred.astype(np.float64) - daac_pred),
    ]
    return np.vstack(cols).T


def run_classification_dataset(
    dataset_name: str,
    train_texts: List[str],
    train_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    seeds: Sequence[int],
    n_partitions: int,
    n_trusted: int,
    poison_flip_prob: float,
    trusted_score: float,
    untrusted_score: float,
    backbone: str,
    tfidf_max_features: int,
    cobra_grid: Sequence[CobraConfig],
    trim_ratio: float,
    mom_groups: int,
    huber_c: float,
) -> Dict[str, Any]:
    cobra_runs: List[float] = []
    daac_runs: List[float] = []
    fusion_runs: List[float] = []
    chosen_algos: List[str] = []
    chosen_robust: List[str] = []

    for seed in seeds:
        x_rm, y_rm, x_meta, y_meta, x_val, y_val = split_three_way(
            train_texts, train_labels, seed=seed, test_size=0.15, val_size=0.15
        )

        models, trusts, _ = build_partition_models(
            texts=x_rm,
            labels=y_rm,
            n_partitions=n_partitions,
            n_trusted=n_trusted,
            poison_flip_prob=poison_flip_prob,
            trusted_score=trusted_score,
            untrusted_score=untrusted_score,
            seed=seed,
            backbone=backbone,
            tfidf_max_features=tfidf_max_features,
        )

        val_prob = score_matrix(models, x_val)
        variances = variance_profile_from_prob(val_prob)
        best_cfg, _ = select_best_cobra_config(
            val_prob=val_prob,
            val_labels=y_val,
            trusts=trusts,
            variances=variances,
            grid=cobra_grid,
            trim_ratio=trim_ratio,
            mom_groups=mom_groups,
            huber_c=huber_c,
        )
        chosen_algos.append(best_cfg.algorithm)
        chosen_robust.append(best_cfg.robust_method)

        # COBRA on test
        test_prob = score_matrix(models, test_texts)
        cobra_pred_test, cobra_conf_test, cobra_score_test = cobra_predict(
            prob_mat=test_prob,
            trusts=trusts,
            variances=variances,
            cfg=best_cfg,
            trim_ratio=trim_ratio,
            mom_groups=mom_groups,
            huber_c=huber_c,
        )
        cobra_runs.append(accuracy(test_labels, cobra_pred_test))

        # DAAC
        meta_prob = score_matrix(models, x_meta)
        x_daac_train = daac_features_from_prob(meta_prob, trusts)
        daac = LogisticRegression(max_iter=3000, random_state=seed)
        daac.fit(x_daac_train, y_meta)

        x_daac_test = daac_features_from_prob(test_prob, trusts)
        daac_prob_test = daac.predict_proba(x_daac_test)[:, 1]
        daac_pred_test = (daac_prob_test >= 0.5).astype(np.int64)
        daac_runs.append(accuracy(test_labels, daac_pred_test))

        # Learned fusion gate (train on val)
        x_daac_val = daac_features_from_prob(val_prob, trusts)
        daac_prob_val = daac.predict_proba(x_daac_val)[:, 1]
        cobra_pred_val, cobra_conf_val, cobra_score_val = cobra_predict(
            prob_mat=val_prob,
            trusts=trusts,
            variances=variances,
            cfg=best_cfg,
            trim_ratio=trim_ratio,
            mom_groups=mom_groups,
            huber_c=huber_c,
        )

        x_gate_train = gate_features(
            prob_mat=val_prob,
            cobra_pred=cobra_pred_val,
            cobra_conf=cobra_conf_val,
            cobra_score=cobra_score_val,
            daac_prob=daac_prob_val,
        )
        gate = LogisticRegression(max_iter=3000, random_state=seed, class_weight="balanced")
        gate.fit(x_gate_train, y_val)

        x_gate_test = gate_features(
            prob_mat=test_prob,
            cobra_pred=cobra_pred_test,
            cobra_conf=cobra_conf_test,
            cobra_score=cobra_score_test,
            daac_prob=daac_prob_test,
        )
        fusion_pred = gate.predict(x_gate_test)
        fusion_runs.append(accuracy(test_labels, fusion_pred))

    result = {
        "dataset": dataset_name,
        "metric": "accuracy",
        "cobra": summarize_runs(cobra_runs),
        "daac": summarize_runs(daac_runs),
        "cobra_plus_daac": summarize_runs(fusion_runs),
        "pvalue_daac_vs_cobra": paired_pvalue(daac_runs, cobra_runs),
        "pvalue_fusion_vs_daac": paired_pvalue(fusion_runs, daac_runs),
        "pvalue_fusion_vs_cobra": paired_pvalue(fusion_runs, cobra_runs),
        "selected_cobra_algorithm_counts": {k: chosen_algos.count(k) for k in ["rot", "drwa", "avga"]},
        "selected_robust_method_counts": {k: chosen_robust.count(k) for k in sorted(set(chosen_robust))},
    }
    return result


def run_hh_rlhf_dataset(
    train_pairs: List[Tuple[str, str]],
    test_pairs: List[Tuple[str, str]],
    seeds: Sequence[int],
    n_partitions: int,
    n_trusted: int,
    poison_flip_prob: float,
    trusted_score: float,
    untrusted_score: float,
    backbone: str,
    tfidf_max_features: int,
    cobra_grid: Sequence[CobraConfig],
    trim_ratio: float,
    mom_groups: int,
    huber_c: float,
) -> Dict[str, Any]:
    cobra_runs: List[float] = []
    daac_runs: List[float] = []
    fusion_runs: List[float] = []
    chosen_algos: List[str] = []
    chosen_robust: List[str] = []

    pair_idx = np.arange(len(train_pairs))

    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(pair_idx)
        n_total = len(perm)
        n_rm = int(0.70 * n_total)
        n_meta = int(0.15 * n_total)
        rm_idx = perm[:n_rm]
        meta_idx = perm[n_rm:n_rm + n_meta]
        val_idx = perm[n_rm + n_meta:]

        rm_pairs = [train_pairs[i] for i in rm_idx]
        rm_texts = [c for c, _ in rm_pairs] + [r for _, r in rm_pairs]
        rm_labels = np.concatenate([np.ones(len(rm_pairs), dtype=np.int64), np.zeros(len(rm_pairs), dtype=np.int64)])

        models, trusts, _ = build_partition_models(
            texts=rm_texts,
            labels=rm_labels,
            n_partitions=n_partitions,
            n_trusted=n_trusted,
            poison_flip_prob=poison_flip_prob,
            trusted_score=trusted_score,
            untrusted_score=untrusted_score,
            seed=seed,
            backbone=backbone,
            tfidf_max_features=tfidf_max_features,
        )

        # COBRA config search on positive val orientation
        val_pairs = [train_pairs[i] for i in val_idx]
        val_chosen = [c for c, _ in val_pairs]
        val_rejected = [r for _, r in val_pairs]
        val_delta = score_matrix(models, val_chosen) - score_matrix(models, val_rejected)
        val_prob_pos = sigmoid(val_delta)
        variances = variance_profile_from_prob(val_prob_pos)
        y_val_pos = np.ones(len(val_pairs), dtype=np.int64)

        best_cfg, _ = select_best_cobra_config(
            val_prob=val_prob_pos,
            val_labels=y_val_pos,
            trusts=trusts,
            variances=variances,
            grid=cobra_grid,
            trim_ratio=trim_ratio,
            mom_groups=mom_groups,
            huber_c=huber_c,
        )
        chosen_algos.append(best_cfg.algorithm)
        chosen_robust.append(best_cfg.robust_method)

        # COBRA test
        test_chosen = [c for c, _ in test_pairs]
        test_rejected = [r for _, r in test_pairs]
        test_delta = score_matrix(models, test_chosen) - score_matrix(models, test_rejected)
        test_prob_pos = sigmoid(test_delta)
        y_test_pos = np.ones(len(test_pairs), dtype=np.int64)

        cobra_pred_test, cobra_conf_test, cobra_score_test = cobra_predict(
            prob_mat=test_prob_pos,
            trusts=trusts,
            variances=variances,
            cfg=best_cfg,
            trim_ratio=trim_ratio,
            mom_groups=mom_groups,
            huber_c=huber_c,
        )
        cobra_runs.append(accuracy(y_test_pos, cobra_pred_test))

        # DAAC pairwise ranking (meta)
        meta_pairs = [train_pairs[i] for i in meta_idx]
        meta_chosen = [c for c, _ in meta_pairs]
        meta_rejected = [r for _, r in meta_pairs]
        meta_delta = score_matrix(models, meta_chosen) - score_matrix(models, meta_rejected)

        x_pos = daac_features_from_pair_delta(meta_delta, trusts)
        x_neg = daac_features_from_pair_delta(-meta_delta, trusts)
        x_rank = np.vstack([x_pos, x_neg])
        y_rank = np.concatenate([
            np.ones(len(meta_pairs), dtype=np.int64),
            np.zeros(len(meta_pairs), dtype=np.int64),
        ])

        # Pairwise ranking as binary orientation classification on (delta, -delta).
        # Logistic model is empirically more stable across seeds than hinge here.
        daac_ranker = LogisticRegression(max_iter=4000, random_state=seed, class_weight="balanced")
        daac_ranker.fit(x_rank, y_rank)

        x_test_pos = daac_features_from_pair_delta(test_delta, trusts)
        daac_margin_test = daac_ranker.decision_function(x_test_pos)
        daac_prob_test = daac_ranker.predict_proba(x_test_pos)[:, 1]
        daac_pred_test = (daac_margin_test >= 0.0).astype(np.int64)
        daac_runs.append(accuracy(y_test_pos, daac_pred_test))

        # Learned fusion gate (val pos/neg)
        x_val_pos = daac_features_from_pair_delta(val_delta, trusts)
        x_val_neg = daac_features_from_pair_delta(-val_delta, trusts)
        daac_prob_val_pos = daac_ranker.predict_proba(x_val_pos)[:, 1]
        daac_prob_val_neg = daac_ranker.predict_proba(x_val_neg)[:, 1]

        cobra_pred_val_pos, cobra_conf_val_pos, cobra_score_val_pos = cobra_predict(
            prob_mat=val_prob_pos,
            trusts=trusts,
            variances=variances,
            cfg=best_cfg,
            trim_ratio=trim_ratio,
            mom_groups=mom_groups,
            huber_c=huber_c,
        )
        val_prob_neg = 1.0 - val_prob_pos
        cobra_pred_val_neg, cobra_conf_val_neg, cobra_score_val_neg = cobra_predict(
            prob_mat=val_prob_neg,
            trusts=trusts,
            variances=variances,
            cfg=best_cfg,
            trim_ratio=trim_ratio,
            mom_groups=mom_groups,
            huber_c=huber_c,
        )

        gate_pos = gate_features(
            prob_mat=val_prob_pos,
            cobra_pred=cobra_pred_val_pos,
            cobra_conf=cobra_conf_val_pos,
            cobra_score=cobra_score_val_pos,
            daac_prob=daac_prob_val_pos,
        )
        gate_neg = gate_features(
            prob_mat=val_prob_neg,
            cobra_pred=cobra_pred_val_neg,
            cobra_conf=cobra_conf_val_neg,
            cobra_score=cobra_score_val_neg,
            daac_prob=daac_prob_val_neg,
        )
        x_gate_train = np.vstack([gate_pos, gate_neg])
        y_gate_train = np.concatenate([
            np.ones(len(gate_pos), dtype=np.int64),
            np.zeros(len(gate_neg), dtype=np.int64),
        ])

        gate = LogisticRegression(max_iter=3000, random_state=seed, class_weight="balanced")
        gate.fit(x_gate_train, y_gate_train)

        gate_test = gate_features(
            prob_mat=test_prob_pos,
            cobra_pred=cobra_pred_test,
            cobra_conf=cobra_conf_test,
            cobra_score=cobra_score_test,
            daac_prob=daac_prob_test,
        )
        fusion_pred = gate.predict(gate_test)
        fusion_runs.append(accuracy(y_test_pos, fusion_pred))

    result = {
        "dataset": "hh_rlhf",
        "metric": "correct_score_accuracy",
        "cobra": summarize_runs(cobra_runs),
        "daac": summarize_runs(daac_runs),
        "cobra_plus_daac": summarize_runs(fusion_runs),
        "pvalue_daac_vs_cobra": paired_pvalue(daac_runs, cobra_runs),
        "pvalue_fusion_vs_daac": paired_pvalue(fusion_runs, daac_runs),
        "pvalue_fusion_vs_cobra": paired_pvalue(fusion_runs, cobra_runs),
        "selected_cobra_algorithm_counts": {k: chosen_algos.count(k) for k in ["rot", "drwa", "avga"]},
        "selected_robust_method_counts": {k: chosen_robust.count(k) for k in sorted(set(chosen_robust))},
    }
    return result


def build_markdown_table(rows: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("| Dataset | Metric | COBRA | DAAC | COBRA+DAAC(gated) | p (DAAC vs COBRA) | p (Fusion vs DAAC) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            "| {dataset} | {metric} | {cobra} | {daac} | {fusion} | {p1:.4g} | {p2:.4g} |".format(
                dataset=row["dataset"],
                metric=row["metric"],
                cobra=fmt_mean_std(row["cobra"]["mean"], row["cobra"]["std"]),
                daac=fmt_mean_std(row["daac"]["mean"], row["daac"]["std"]),
                fusion=fmt_mean_std(row["cobra_plus_daac"]["mean"], row["cobra_plus_daac"]["std"]),
                p1=row["pvalue_daac_vs_cobra"],
                p2=row["pvalue_fusion_vs_daac"],
            )
        )

    cobra_means = [r["cobra"]["mean"] for r in rows]
    daac_means = [r["daac"]["mean"] for r in rows]
    fusion_means = [r["cobra_plus_daac"]["mean"] for r in rows]
    lines.append(
        "| macro_avg | mean(metric) | {c} | {d} | {f} | {p1:.4g} | {p2:.4g} |".format(
            c=fmt_mean_std(float(np.mean(cobra_means)), float(np.std(cobra_means, ddof=1) if len(cobra_means) > 1 else 0.0)),
            d=fmt_mean_std(float(np.mean(daac_means)), float(np.std(daac_means, ddof=1) if len(daac_means) > 1 else 0.0)),
            f=fmt_mean_std(float(np.mean(fusion_means)), float(np.std(fusion_means, ddof=1) if len(fusion_means) > 1 else 0.0)),
            p1=paired_pvalue(daac_means, cobra_means),
            p2=paired_pvalue(fusion_means, daac_means),
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="COBRA paper-style comparison with full COBRA/DAAC upgrades")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="comma-separated seeds")

    parser.add_argument("--n-partitions", type=int, default=15)
    parser.add_argument("--trusted-ratio", type=float, default=0.6)
    parser.add_argument("--poison-flip-prob", type=float, default=0.1)
    parser.add_argument("--trusted-score", type=float, default=0.85)
    parser.add_argument("--untrusted-score", type=float, default=0.45)

    parser.add_argument("--backbone", choices=["tfidf", "hashing", "char_ngram", "naive_bayes", "diverse"], default="tfidf")
    parser.add_argument("--tfidf-max-features", type=int, default=70000)

    parser.add_argument("--cobra-robust-methods", type=str, default="standard,trimmed,mom,huber")
    parser.add_argument("--rot-thresholds", type=str, default="0.6,0.7,0.8")
    parser.add_argument("--rot-alphas", type=str, default="0.2,0.35,0.5")
    parser.add_argument("--drwa-epsilons", type=str, default="0.1,0.25,0.4")
    parser.add_argument("--avga-betas", type=str, default="0.8,1.5,2.5")

    parser.add_argument("--trim-ratio", type=float, default=0.1)
    parser.add_argument("--mom-groups", type=int, default=5)
    parser.add_argument("--huber-c", type=float, default=1.5)

    parser.add_argument("--max-sst2-train", type=int, default=24000)
    parser.add_argument("--max-imdb-train", type=int, default=20000)
    parser.add_argument("--max-imdb-test", type=int, default=8000)
    parser.add_argument("--max-hh-train-pairs", type=int, default=25000)
    parser.add_argument("--max-hh-test-pairs", type=int, default=4000)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/cobra_paper_style_comparison"),
    )
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    robust_methods = parse_str_list(args.cobra_robust_methods)
    rot_thresholds = parse_float_list(args.rot_thresholds)
    rot_alphas = parse_float_list(args.rot_alphas)
    drwa_epsilons = parse_float_list(args.drwa_epsilons)
    avga_betas = parse_float_list(args.avga_betas)

    cobra_grid = build_cobra_grid(
        robust_methods=robust_methods,
        rot_thresholds=rot_thresholds,
        rot_alphas=rot_alphas,
        drwa_epsilons=drwa_epsilons,
        avga_betas=avga_betas,
    )

    n_trusted = max(1, min(args.n_partitions - 1, int(round(args.n_partitions * args.trusted_ratio))))
    rng = np.random.default_rng(12345)

    # SST-2
    sst2 = load_dataset("sst2")
    sst2_train_texts = list(sst2["train"]["sentence"])
    sst2_train_labels = np.array(sst2["train"]["label"], dtype=np.int64)
    # GLUE SST-2 test labels are hidden(-1), use validation as labeled eval split.
    sst2_test_texts = list(sst2["validation"]["sentence"])
    sst2_test_labels = np.array(sst2["validation"]["label"], dtype=np.int64)
    sst2_train_texts, sst2_train_labels = sample_rows(sst2_train_texts, sst2_train_labels, args.max_sst2_train, rng)

    sst2_result = run_classification_dataset(
        dataset_name="sst2",
        train_texts=sst2_train_texts,
        train_labels=sst2_train_labels,
        test_texts=sst2_test_texts,
        test_labels=sst2_test_labels,
        seeds=seeds,
        n_partitions=args.n_partitions,
        n_trusted=n_trusted,
        poison_flip_prob=args.poison_flip_prob,
        trusted_score=args.trusted_score,
        untrusted_score=args.untrusted_score,
        backbone=args.backbone,
        tfidf_max_features=args.tfidf_max_features,
        cobra_grid=cobra_grid,
        trim_ratio=args.trim_ratio,
        mom_groups=args.mom_groups,
        huber_c=args.huber_c,
    )

    # IMDB
    imdb = load_dataset("imdb")
    imdb_train_texts = list(imdb["train"]["text"])
    imdb_train_labels = np.array(imdb["train"]["label"], dtype=np.int64)
    imdb_test_texts = list(imdb["test"]["text"])
    imdb_test_labels = np.array(imdb["test"]["label"], dtype=np.int64)
    imdb_train_texts, imdb_train_labels = sample_rows(imdb_train_texts, imdb_train_labels, args.max_imdb_train, rng)
    imdb_test_texts, imdb_test_labels = sample_rows(imdb_test_texts, imdb_test_labels, args.max_imdb_test, rng)

    imdb_result = run_classification_dataset(
        dataset_name="imdb",
        train_texts=imdb_train_texts,
        train_labels=imdb_train_labels,
        test_texts=imdb_test_texts,
        test_labels=imdb_test_labels,
        seeds=seeds,
        n_partitions=args.n_partitions,
        n_trusted=n_trusted,
        poison_flip_prob=args.poison_flip_prob,
        trusted_score=args.trusted_score,
        untrusted_score=args.untrusted_score,
        backbone=args.backbone,
        tfidf_max_features=args.tfidf_max_features,
        cobra_grid=cobra_grid,
        trim_ratio=args.trim_ratio,
        mom_groups=args.mom_groups,
        huber_c=args.huber_c,
    )

    # HH-RLHF
    hh = load_dataset("Anthropic/hh-rlhf")
    hh_train_pairs = list(zip(hh["train"]["chosen"], hh["train"]["rejected"]))
    hh_test_pairs = list(zip(hh["test"]["chosen"], hh["test"]["rejected"]))

    if args.max_hh_train_pairs > 0 and len(hh_train_pairs) > args.max_hh_train_pairs:
        idx = rng.choice(len(hh_train_pairs), size=args.max_hh_train_pairs, replace=False)
        idx.sort()
        hh_train_pairs = [hh_train_pairs[i] for i in idx]
    if args.max_hh_test_pairs > 0 and len(hh_test_pairs) > args.max_hh_test_pairs:
        idx = rng.choice(len(hh_test_pairs), size=args.max_hh_test_pairs, replace=False)
        idx.sort()
        hh_test_pairs = [hh_test_pairs[i] for i in idx]

    hh_result = run_hh_rlhf_dataset(
        train_pairs=hh_train_pairs,
        test_pairs=hh_test_pairs,
        seeds=seeds,
        n_partitions=args.n_partitions,
        n_trusted=n_trusted,
        poison_flip_prob=args.poison_flip_prob,
        trusted_score=args.trusted_score,
        untrusted_score=args.untrusted_score,
        backbone=args.backbone,
        tfidf_max_features=args.tfidf_max_features,
        cobra_grid=cobra_grid,
        trim_ratio=args.trim_ratio,
        mom_groups=args.mom_groups,
        huber_c=args.huber_c,
    )

    results = [sst2_result, imdb_result, hh_result]
    table_md = build_markdown_table(results)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"cobra_paper_style_comparison_upgraded_{ts}.json"
    md_path = out_dir / f"cobra_paper_style_table_upgraded_{ts}.md"

    payload = {
        "timestamp_utc": ts,
        "config": {
            "seeds": seeds,
            "n_partitions": args.n_partitions,
            "n_trusted": n_trusted,
            "poison_flip_prob": args.poison_flip_prob,
            "trusted_score": args.trusted_score,
            "untrusted_score": args.untrusted_score,
            "backbone": args.backbone,
            "tfidf_max_features": args.tfidf_max_features,
            "cobra_robust_methods": robust_methods,
            "rot_thresholds": rot_thresholds,
            "rot_alphas": rot_alphas,
            "drwa_epsilons": drwa_epsilons,
            "avga_betas": avga_betas,
            "trim_ratio": args.trim_ratio,
            "mom_groups": args.mom_groups,
            "huber_c": args.huber_c,
            "max_sst2_train": args.max_sst2_train,
            "max_imdb_train": args.max_imdb_train,
            "max_imdb_test": args.max_imdb_test,
            "max_hh_train_pairs": args.max_hh_train_pairs,
            "max_hh_test_pairs": args.max_hh_test_pairs,
        },
        "results": results,
        "table_markdown": table_md,
        "notes": [
            "Upgrades applied: stronger shared RM backbone, COBRA hyperparameter search, robust aggregation, DAAC pairwise ranking, learned gated fusion.",
            "SST-2 uses validation split for labeled evaluation because test labels are hidden on GLUE.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(table_md + "\n", encoding="utf-8")

    print(table_md)
    print(f"\nJSON: {json_path}")
    print(f"TABLE: {md_path}")


if __name__ == "__main__":
    main()
