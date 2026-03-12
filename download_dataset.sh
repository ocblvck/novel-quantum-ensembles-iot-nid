#!/bin/bash

cat <<'EOF'
IoTID20 dataset download

This repository expects the dataset file to be named:
IoT_Original_Distribution.csv

Download sources:

1. Kaggle
	https://www.kaggle.com/datasets/subhajournal/iotid20-iot-botnet-dataset

2. IEEE DataPort
	https://ieee-dataport.org/open-access/iot-network-intrusion-dataset

If you use the Kaggle CLI:

  pip install kaggle
  kaggle datasets download -d subhajournal/iotid20-iot-botnet-dataset
  unzip iotid20-iot-botnet-dataset.zip
  mv 'IoT Network Intrusion Dataset.csv' IoT_Original_Distribution.csv

Place the renamed CSV in this directory before running the script.
EOF
