from itertools import product

import numpy as np

from src.ensemble import DatabaseBundle, compute_three_model_scores
from src.factory import build_confidence_router
from src.transforms import transform_tta_mega, transforms_aliked


def split_for_router_tuning(dataset, val_ratio=0.3, seed=42):
    n = len(dataset)
    if n < 4:
        full_mask = np.ones(n, dtype=bool)
        return full_mask, full_mask

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    val_size = max(1, int(n * val_ratio))
    val_idx = indices[:val_size]
    db_idx = indices[val_size:]

    if len(db_idx) == 0:
        db_idx = indices[:-1]
        val_idx = indices[-1:]

    db_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    db_mask[db_idx] = True
    val_mask[val_idx] = True
    return db_mask, val_mask


def evaluate_top1_closed_set(scores, query_dataset, db_dataset):
    top_idx = scores.argmax(axis=1)
    pred = db_dataset.labels_string[top_idx]
    gt = query_dataset.labels_string

    known_mask = np.isin(gt, db_dataset.labels_string)
    if known_mask.sum() == 0:
        return 0.0
    return float((pred[known_mask] == gt[known_mask]).mean())


def generate_weight_candidates(weight_step):
    values = np.round(np.arange(weight_step, 1.0, weight_step), 4)
    weight_candidates = []
    for w_mega, w_aliked in product(values, values):
        w_eva = 1.0 - w_mega - w_aliked
        if w_eva < weight_step:
            continue
        weight_candidates.append(
            {
                "mega": float(round(w_mega, 4)),
                "aliked": float(round(w_aliked, 4)),
                "eva": float(round(w_eva, 4)),
            }
        )
    return weight_candidates


def tune_router_on_calib(
    dataset_calib,
    pipelines,
    default_base_weights,
    default_confidence_mix,
    val_ratio,
    topk_for_aliked,
    weight_step,
    mix_candidates,
):
    print("[RouterTune] Preparing calibration split for router tuning...")
    db_mask, val_mask = split_for_router_tuning(
        dataset_calib,
        val_ratio=val_ratio,
        seed=42,
    )
    tune_db_base = dataset_calib.get_subset(db_mask)
    tune_query_base = dataset_calib.get_subset(val_mask)

    db_full_mask = np.ones(len(tune_db_base), dtype=bool)
    query_full_mask = np.ones(len(tune_query_base), dtype=bool)

    db_mega = tune_db_base.get_subset(db_full_mask)
    db_mega.transform = transform_tta_mega
    db_aliked = tune_db_base.get_subset(db_full_mask)
    db_aliked.transform = transforms_aliked
    db_eva = tune_db_base.get_subset(db_full_mask)
    db_eva.transform = pipelines.eva.transform
    db_bundle = DatabaseBundle(mega=db_mega, aliked=db_aliked, eva=db_eva)

    query_subset = tune_query_base.get_subset(query_full_mask)
    print("[RouterTune] Computing model scores on tuning split...")
    scores_mega, scores_aliked_full, scores_eva = compute_three_model_scores(
        query_subset=query_subset,
        db_bundle=db_bundle,
        pipelines=pipelines,
        topk_for_aliked=topk_for_aliked,
    )

    weight_candidates = generate_weight_candidates(weight_step)
    if not weight_candidates:
        weight_candidates = [default_base_weights]

    valid_mask = np.isfinite(scores_aliked_full)
    best_score = -1.0
    best_base = default_base_weights
    best_mix = default_confidence_mix

    print(
        f"[RouterTune] Searching {len(weight_candidates)} weight combos x "
        f"{len(mix_candidates)} mix values..."
    )
    for base_weights in weight_candidates:
        for mix in mix_candidates:
            router = build_confidence_router(base_weights=base_weights, confidence_mix=mix)
            fused_scores = router.fuse(
                scores_mega=scores_mega,
                scores_aliked=scores_aliked_full,
                scores_eva=scores_eva,
                aliked_valid_mask=valid_mask,
            )
            top1_acc = evaluate_top1_closed_set(
                scores=fused_scores,
                query_dataset=query_subset,
                db_dataset=tune_db_base,
            )
            if top1_acc > best_score:
                best_score = top1_acc
                best_base = base_weights
                best_mix = float(mix)

    print(
        f"[RouterTune] Best closed-set top1={best_score:.4f} "
        f"with base_weights={best_base}, confidence_mix={best_mix}"
    )
    return best_base, best_mix
