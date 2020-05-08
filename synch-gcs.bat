@echo off
set GCS_BUCKET=gs://eo-ald-update
set ROOT_PATH=c:\Users\Chris.Williams\Documents\GitHub
set REPO=cowc

rem gsutil cp %ROOT_PATH%\%REPO%\data\train.zip %GCS_BUCKET%/%REPO%/data/train.zip
rem gsutil cp %ROOT_PATH%\%REPO%\data\test.zip %GCS_BUCKET%/%REPO%/data/test.zip
gsutil cp %ROOT_PATH%\%REPO%\models\yolo3-default.zip %GCS_BUCKET%/%REPO%/models/yolo3-default.zip
