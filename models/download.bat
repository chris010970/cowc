@echo off
set GCS_BUCKET=gs://eo-ald-update
set REPO=cowc

gsutil cp %GCS_BUCKET%/%REPO%/models/yolo3-default/cowc.zip .
