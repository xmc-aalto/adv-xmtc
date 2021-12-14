#!/bin/bash

cur_dir=$(pwd)

data_path=data_defense/data/aplc/Wiki10
mkdir -p $data_path
echo "Downloading the Wikipedia-31K dataset"
cd $data_path
dataid="1iBji8M5QpUX4KPyHKJ_QgH-r1-2Q_7bs"
dataname="wiki10.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${dataid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${dataid}" -o ${dataname}
tar -xzvf $dataname
rm $dataname
rm ./cookie
cd $cur_dir


model_path=models/Wiki10/aplc/None
mkdir -p $model_path
echo "Downloading APLC_XlNet model trained on Wikipedia-31K"
cd $model_path
modelid="1-O0tD_MjD4QDM32xWzxakaSC-S1DfTeU"
modelname="wiki10.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${modelid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${modelid}" -o ${modelname}
tar -xzvf $modelname
rm $modelname
rm ./cookie

