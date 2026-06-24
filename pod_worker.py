#!/usr/bin/env python3
"""
RunPod Pod worker — polls Firestore for runpod-pod jobs (hybrid baseline GPU).

Env:
  GOOGLE_APPLICATION_CREDENTIALS=/path/service-account.json
  FIREBASE_PROJECT_ID=super-ai-video-7ddce
  FIREBASE_STORAGE_BUCKET=super-ai-video-7ddce.firebasestorage.app
  WORKER_ID=runpod-pod-1
  POLL_INTERVAL_SEC=5
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import bootstrap  # noqa: F401

import firebase_admin
from firebase_admin import credentials, firestore, storage

PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "super-ai-video-7ddce").strip()
STORAGE_BUCKET = os.environ.get(
    "FIREBASE_STORAGE_BUCKET", f"{PROJECT_ID}.firebasestorage.app"
).strip()
POLL_SEC = float(os.environ.get("POLL_INTERVAL_SEC", "5"))
WORKER_ID = os.environ.get("WORKER_ID", "runpod-pod-1")
POD_PROVIDER = "runpod-pod"


def _log(msg: str) -> None:
    print(msg, flush=True)


def init_firebase() -> firestore.Client:
    if not firebase_admin._apps:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if cred_path and Path(cred_path).is_file():
            cred = credentials.Certificate(cred_path)
        else:
            cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {"storageBucket": STORAGE_BUCKET})
    return firestore.client()


def report_status(db: firestore.Client) -> None:
    cuda = False
    gpu_name = ""
    mode = "init"
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        if cuda:
            gpu_name = str(torch.cuda.get_device_name(0))
            mode = "wan-14b"
    except Exception as e:
        mode = f"error:{e}"[:120]

    db.collection("runpod_worker_status").doc("current").set(
        {
            "workerId": WORKER_ID,
            "cudaAvailable": cuda,
            "gpuName": gpu_name,
            "generationMode": mode,
            "provider": POD_PROVIDER,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )


def claim_next_job(db: firestore.Client) -> firestore.DocumentSnapshot | None:
    q = (
        db.collection("video_jobs")
        .where("status", "==", "queued")
        .order_by("createdAt")
        .limit(20)
    )
    for doc in q.stream():
        if doc.get("provider") != POD_PROVIDER:
            continue
        ref = doc.reference

        @firestore.transactional
        def claim_in_tx(transaction: firestore.Transaction) -> firestore.DocumentSnapshot | None:
            snap = ref.get(transaction=transaction)
            if not snap.exists or snap.get("status") != "queued":
                return None
            if snap.get("provider") != POD_PROVIDER:
                return None
            transaction.update(
                ref,
                {
                    "status": "processing",
                    "workerId": WORKER_ID,
                    "startedAt": firestore.SERVER_TIMESTAMP,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
            )
            return snap

        claimed = claim_in_tx(db.transaction())
        if claimed is not None:
            return claimed
    return None


def upload_output(uid: str, job_id: str, local_mp4: Path) -> str:
    bucket = storage.bucket()
    blob_path = f"video_outputs/{uid}/{job_id}.mp4"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_mp4), content_type="video/mp4")
    blob.make_public()
    return blob.public_url


def process_job(db: firestore.Client, snap: firestore.DocumentSnapshot) -> None:
    from wan_engine import WAN_MODEL_ID, generate_video
    from watermark_ffmpeg import burn_animated_watermark

    ref = snap.reference
    data = snap.to_dict() or {}
    prompt = str(data.get("prompt", "")).strip()
    duration_sec = int(data.get("durationSec") or 4)
    uid = str(data.get("uid", ""))
    job_id = snap.id
    watermark = bool(data.get("watermark"))
    watermark_spec = data.get("watermarkSpec") or {}
    generation_mode = str(data.get("generationMode") or "standard").strip().lower()
    ad_category = str(data.get("adCategory") or "auto").strip().lower()
    product_image_url = str(data.get("productImageUrl") or "").strip() or None

    _log(f"[pod_worker] job={job_id} uid={uid} dur={duration_sec}s mode={generation_mode}")

    with tempfile.TemporaryDirectory(prefix="pod_video_") as tmp:
        tmp_dir = Path(tmp)
        raw_mp4 = tmp_dir / "raw.mp4"
        final_mp4 = tmp_dir / "final.mp4"

        generate_video(
            prompt,
            duration_sec,
            raw_mp4,
            model_id=WAN_MODEL_ID,
            generation_mode=generation_mode,
            ad_category=ad_category,
            product_image_url=product_image_url,
        )
        out_path = final_mp4 if watermark else raw_mp4
        if watermark:
            burn_animated_watermark(raw_mp4, final_mp4, watermark_spec)

        url = upload_output(uid, job_id, out_path)
        ref.update(
            {
                "status": "done",
                "outputUrl": url,
                "model": WAN_MODEL_ID,
                "finishedAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            }
        )
        _log(f"[pod_worker] done → {url}")


def fail_job(ref: firestore.DocumentReference, message: str) -> None:
    ref.update(
        {
            "status": "failed",
            "error": message[:500],
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
    )


def main() -> None:
    _log(f"[pod_worker] starting {WORKER_ID} project={PROJECT_ID}")
    db = init_firebase()

    from wan_engine import warmup

    warmup()

    while True:
        try:
            report_status(db)
            snap = claim_next_job(db)
            if snap is None:
                time.sleep(POLL_SEC)
                continue
            try:
                process_job(db, snap)
            except Exception as e:
                traceback.print_exc()
                fail_job(snap.reference, str(e))
        except KeyboardInterrupt:
            _log("[pod_worker] stopped")
            sys.exit(0)
        except Exception as e:
            traceback.print_exc()
            _log(f"[pod_worker] loop error: {e}")
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
