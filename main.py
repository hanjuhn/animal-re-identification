import pandas as pd

from config import (
    ROOT,
    MEGAD_NAME,
    DEVICE,
    THRESHOLD,
    ROUTER_BASE_WEIGHTS,
    ROUTER_CONFIDENCE_MIX,
    ROUTER_AUTO_TUNE,
    ROUTER_TUNE_VAL_RATIO,
    ROUTER_TUNE_TOPK,
    ROUTER_TUNE_WEIGHT_STEP,
    ROUTER_TUNE_MIX_CANDIDATES,
)
from src.dataset import load_datasets
from src.ensemble import (
    build_pipelines,
    fit_calibrators,
    make_database_bundle,
    run_inference_by_dataset,
)
from src.factory import build_confidence_router
from src.router_tuning import tune_router_on_calib
from src.utils import set_seed


def main():
    set_seed(42)

    print("[Main] Loading datasets...")
    dataset, dataset_db, dataset_query, dataset_calib = load_datasets(ROOT, calibration_size=1000)

    pipelines = build_pipelines(MEGAD_NAME, DEVICE)
    fit_calibrators(dataset_calib, pipelines)

    router_base_weights = ROUTER_BASE_WEIGHTS
    router_confidence_mix = ROUTER_CONFIDENCE_MIX
    if ROUTER_AUTO_TUNE:
        router_base_weights, router_confidence_mix = tune_router_on_calib(
            dataset_calib=dataset_calib,
            pipelines=pipelines,
            default_base_weights=ROUTER_BASE_WEIGHTS,
            default_confidence_mix=ROUTER_CONFIDENCE_MIX,
            val_ratio=ROUTER_TUNE_VAL_RATIO,
            topk_for_aliked=ROUTER_TUNE_TOPK,
            weight_step=ROUTER_TUNE_WEIGHT_STEP,
            mix_candidates=ROUTER_TUNE_MIX_CANDIDATES,
        )

    router = build_confidence_router(
        base_weights=router_base_weights,
        confidence_mix=router_confidence_mix,
    )
    print(
        f"[Main] Router config: base_weights={router_base_weights}, "
        f"confidence_mix={router_confidence_mix}"
    )

    db_bundle = make_database_bundle(dataset_db, pipelines)
    image_ids_all, predictions_all = run_inference_by_dataset(
        dataset_query=dataset_query,
        dataset_db=dataset_db,
        db_bundle=db_bundle,
        pipelines=pipelines,
        router=router,
        threshold=THRESHOLD,
        topk_for_aliked=ROUTER_TUNE_TOPK,
    )

    df = pd.DataFrame({"image_id": image_ids_all, "identity": predictions_all})
    df.to_csv("sample_submission.csv", index=False)
    print("✅ sample_submission.csv saved!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
