#!/bin/bash
curl -L -o ./environmental-sound-classification-50.zip\
  https://www.kaggle.com/api/v1/datasets/download/mmoreaux/environmental-sound-classification-50
unzip -q environmental-sound-classification-50.zip -d ./esc50_mel_cnn/data
rm environmental-sound-classification-50.zip
