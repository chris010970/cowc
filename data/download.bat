@echo off
set GCS_BUCKET=gs://eo-ald-update
set REPO=cowc

gsutil cp %GCS_BUCKET%/%REPO%/data/train.zip .
gsutil cp %GCS_BUCKET%/%REPO%/data/test.zip .
