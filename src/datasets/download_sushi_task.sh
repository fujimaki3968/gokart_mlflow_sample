#!/bin/bash

### 事前準備
# gdownなどのツールを使うよりも確実なので以下の方法を採用する
# https://developers.google.com/oauthplayground/ にアクセスし、Drive API v3を選択
# 認証後にアクセストークンを取得し、このスクリプトの実行時に入力する
#
# pythonはpandas, openpyxlが必要

echo "事前準備"
echo "データセットをダウンロードするためにGoogle Drive APIのアクセストークンが必要です"
echo "1. https://developers.google.com/oauthplayground/ にアクセスし、Drive API v3を選択"
echo "2. 認証後にアクセストークンを取得し、このスクリプトの実行時に入力する"

# アクセストークンの入力を求める
read -p "Google Drive APIのアクセストークンを入力してください: " access_token

target_dir="data/external/sushi"
mkdir -p $target_dir

# curlコマンドを実行

## large file
curl -H "Authorization: Bearer $access_token" \
     https://www.googleapis.com/drive/v3/files/1hA5FW0cNloi20coLlGvnv5wMap8ZN8YL?alt=media \
     -o $target_dir/sushi-files.zip

# zipの解凍
# UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE にしないと、zipファイルの解凍時にエラーが発生する
echo "Downloadしたzipファイルを解凍します"
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE; unzip $target_dir/sushi-files.zip -d $target_dir/ -x "__MACOSX/*"
echo "解凍が完了しました"

curl -H "Authorization: Bearer $access_token" \
     https://www.googleapis.com/drive/v3/files/1-_e-qo9aIhjTPSxhDkkWDlxN2N0Jo4FA?alt=media \
     -o $target_dir/SubtaskACollectionMetadataV1.1.xlsx

curl -H "Authorization: Bearer $access_token" \
     https://www.googleapis.com/drive/v3/files/1vcB9TCa5t8P3IKg4XmGThu5xULYtNm9c?alt=media \
     -o $target_dir/Ntcir18SushiDryRunExperimentControlFileV1.1.json

curl -H "Authorization: Bearer $access_token" \
     https://www.googleapis.com/drive/v3/files/1R3ycNbN6wvxw7V0DE86P3cKYdHCi0Ngh?alt=media \
     -o $target_dir/SncTranslationV1.1.xlsx

curl -H "Authorization: Bearer $access_token" \
     https://www.googleapis.com/drive/v3/files/1ww8eYdS1WXPFQkm8pS0EoDvd6Q7W6Lwb?alt=media \
     -o $target_dir/SubtaskAValidator.py

curl -H "Authorization: Bearer $access_token" \
     https://www.googleapis.com/drive/v3/files/1ZdUdGT4HKYzdxFEn8KyxEFqrq1ypER2O?alt=media \
     -o $target_dir/SubtaskAEvaluation.py

curl -H "Authorization: Bearer $access_token" \
     https://www.googleapis.com/drive/v3/files/1Q6553xByDHcxhqMgrsNDT5O_a9f3VB9x?alt=media \
     -o $target_dir/Ntcir18SushiDryRunFolderQrelsV1.1.tsv

curl -H "Authorization: Bearer $access_token" \
     https://www.googleapis.com/drive/v3/files/1rVwHYOtY-PpG-RUo44H7lY9PmHNwbuL0?alt=media \
     -o $target_dir/Ntcir18SushiDryRunBoxQrelsV1.1.tsv


# 完了メッセージを表示
echo "ダウンロードが完了しました。"
