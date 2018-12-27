# !/bin/bash

NOISY_PATH="data/noisy_trainset_wav_16k"
ENHANCED_PATH="data/enhanced_segan"
SUB_DIR="/*.wav"

mkdir -p $ENHANCED_PATH

FILES=$NOISY_PATH$SUB_DIR
echo $FILES
for f in $FILES; do
    echo $f 
#     clean_wav.sh $f $ENHANCED_PATH
done
