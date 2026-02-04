from dataclasses import dataclass

import numpy as np
import timm
import torch

from src.factory import build_aliked, build_eva02, build_megadescriptor
from src.transforms import transform_tta_mega, transforms_aliked


@dataclass
class PipelineBundle:
    mega: object
    aliked: object
    eva: object


@dataclass
class DatabaseBundle:
    mega: object
    aliked: object
    eva: object


def build_pipelines(megad_name, device):
    print("[Main] Building pipelines...")
    print(f" - MegaDescriptor ({megad_name})")
    model_mega = timm.create_model(megad_name, num_classes=0, pretrained=True).to(device)
    pipeline_mega = build_megadescriptor(
        model=model_mega,
        transform=transform_tta_mega,
        device=device,
    )

    print(" - ALIKED (Local Features)")
    pipeline_aliked = build_aliked(transform=transforms_aliked, device=device)

    print(" - EVA02 (Global Features)")
    pipeline_eva = build_eva02(device=device)

    return PipelineBundle(
        mega=pipeline_mega,
        aliked=pipeline_aliked,
        eva=pipeline_eva,
    )


def fit_calibrators(dataset_calib, pipelines):
    print("[Main] Starting calibration...")
    calib_mask = np.ones(len(dataset_calib), dtype=bool)

    calib_mega = dataset_calib.get_subset(calib_mask)
    calib_mega.transform = transform_tta_mega
    pipelines.mega.fit_calibration(calib_mega, calib_mega)

    calib_aliked = dataset_calib.get_subset(calib_mask)
    calib_aliked.transform = transforms_aliked
    pipelines.aliked.fit_calibration(calib_aliked, calib_aliked)

    calib_eva = dataset_calib.get_subset(calib_mask)
    calib_eva.transform = pipelines.eva.transform
    pipelines.eva.fit_calibration(calib_eva, calib_eva)


def make_database_bundle(dataset_db, pipelines):
    db_mask = np.ones(len(dataset_db), dtype=bool)

    db_mega = dataset_db.get_subset(db_mask)
    db_mega.transform = transform_tta_mega

    db_aliked = dataset_db.get_subset(db_mask)
    db_aliked.transform = transforms_aliked

    db_eva = dataset_db.get_subset(db_mask)
    db_eva.transform = pipelines.eva.transform

    return DatabaseBundle(
        mega=db_mega,
        aliked=db_aliked,
        eva=db_eva,
    )


def compute_three_model_scores(query_subset, db_bundle, pipelines, topk_for_aliked=25):
    subset_full_mask = np.ones(len(query_subset), dtype=bool)

    query_mega = query_subset.get_subset(subset_full_mask)
    query_mega.transform = transform_tta_mega
    scores_mega = pipelines.mega(query_mega, db_bundle.mega)

    topk = min(topk_for_aliked, scores_mega.shape[1])
    _, topk_indices = torch.topk(torch.from_numpy(scores_mega), k=topk, dim=1)
    pairs = []
    rows = np.arange(len(query_subset))
    for r, cols in zip(rows, topk_indices.numpy()):
        for c in cols:
            pairs.append((r, c))

    query_aliked = query_subset.get_subset(subset_full_mask)
    query_aliked.transform = transforms_aliked
    scores_aliked_sparse = pipelines.aliked(query_aliked, db_bundle.aliked, pairs=pairs)

    scores_aliked_full = np.full_like(scores_mega, -np.inf)
    if scores_aliked_sparse.ndim == 1:
        q_idxs = [p[0] for p in pairs]
        db_idxs = [p[1] for p in pairs]
        scores_aliked_full[q_idxs, db_idxs] = scores_aliked_sparse
    else:
        scores_aliked_full = scores_aliked_sparse

    query_eva = query_subset.get_subset(subset_full_mask)
    query_eva.transform = pipelines.eva.transform
    scores_eva = pipelines.eva(query_eva, db_bundle.eva)

    return scores_mega, scores_aliked_full, scores_eva


def run_inference_by_dataset(
    dataset_query,
    dataset_db,
    db_bundle,
    pipelines,
    router,
    threshold,
    topk_for_aliked,
):
    predictions_all = []
    image_ids_all = []

    print("[Main] Processing queries by dataset...")
    for dataset_name in dataset_query.metadata["dataset"].unique():
        query_mask = dataset_query.metadata["dataset"] == dataset_name
        query_subset = dataset_query.get_subset(query_mask)
        print(f" -> Processing {dataset_name} ({len(query_subset)} images)...")

        scores_mega, scores_aliked_full, scores_eva = compute_three_model_scores(
            query_subset=query_subset,
            db_bundle=db_bundle,
            pipelines=pipelines,
            topk_for_aliked=topk_for_aliked,
        )

        valid_mask = np.isfinite(scores_aliked_full)
        final_scores = router.fuse(
            scores_mega=scores_mega,
            scores_aliked=scores_aliked_full,
            scores_eva=scores_eva,
            aliked_valid_mask=valid_mask,
        )

        top_idx = final_scores.argmax(axis=1)
        p_top1 = final_scores[np.arange(len(query_subset)), top_idx]

        pred_labels = dataset_db.labels_string[top_idx].copy()
        pred_labels[p_top1 < threshold] = "new_individual"

        predictions_all.extend(pred_labels)
        image_ids_all.extend(query_subset.metadata["image_id"])

    return image_ids_all, predictions_all
