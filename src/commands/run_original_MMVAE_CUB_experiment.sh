#!/bin/bash

# OUTPUTDIR="../outputs"
# DATADIR="../../../../../Data/Dataset/CUB"
marker="" # "_test" for test, "" for train
DATASET="CUB" # CUB, CUBICC, CUBcluster8, TODO::CUB temp: new_release:1590 -> temp:1546
if [ "$DATASET" = "CUB" ]; then
    DATADIR="/data/backed_up/shared/Data/CUB/CUB_captions_new/CUB_70_10_20"
    EXPERIMENT="CUB_from_May28${marker}"
elif [ "$DATASET" = "CUBICC" ]; then
    DATADIR="/data/backed_up/shared/Data/CUB/CUBICC"
    EXPERIMENT="CUBICC_1${marker}"
elif [ "$DATASET" = "CUBcluster8" ]; then
    # DATADIR="/data/backed_up/shared/Data/CUB/CUBcluster8"
    EXPERIMENT="CUBcluster8_from_May24${marker}"
fi

# TODO (Yijie): add elboVCCA and dregVCCA replacing vcca
MODEL="cubIS"  # mnist_svhn, cubISft, cubIS
OBJ="iwae"  # elbo, dreg, iwae
BATCH=128  # new_release: 32
K=1  # new_release: 10
EPOCHS=200  # new_release: 50
SEED=2  # new_release: 2

# beta=1.0  # new_release: 1.0
# priorposterior="Normal" # Normal, Laplace, Diffusion # new_release: "Normal"
SHARED_LAT_DIM=128  # new_release: 48
# MS_LAT_DIM=0  # new_release: 16

# INDEP_L_SC=0.0
# AUG_L_SC=0.0
# MVIB_L_SC=0.0

date=`echo $(date '+%m-%d')`
number=3
gpuid=3
note=""

# new_release: dreg+Z48+W16+K10+Normal
# --obj "elbo" for ELBO, "dreg" for DReG, "vcca" for VCCA
# --priorposterior "Normal" for Normal prior and posterior, "Laplace" for Laplace prior and posterior
# debug_log and debug_pdb for debugging, default is False, set to True to enable
CUDA_VISIBLE_DEVICES=${gpuid} python main.py --experiment $EXPERIMENT --model $MODEL --obj $OBJ \
      --K $K --batch-size $BATCH --epochs $EPOCHS \
      --latent-dim $SHARED_LAT_DIM --seed $SEED \
      --note "${date}_${number}_gpu${gpuid}${note}"
# 2>&1 | tee -a "${DATADIR}/${DATASET}/logs/log_${EXPERIMENT}_${number}_gpu${gpuid}_B${BATCH}_${AUG_L_SC}_${MVIB_L_SC}_${note}.txt"
    #    --beta $beta --latent-dim-w $MS_LAT_DIM --datadir $DATADIR  --outputdir $OUTPUTDIR \
    #   --dataset $DATASET \
    #   --debug_log --debug_pdb \
    #   --inception_path "${DATADIR}/pt_inception-2015-12-05-6726825d.pth" \
    #   --indep_loss_scale $INDEP_L_SC --augmented_mi_scale $AUG_L_SC --mvib_loss_scale $MVIB_L_SC \
    #   --priorposterior ${priorposterior} \

