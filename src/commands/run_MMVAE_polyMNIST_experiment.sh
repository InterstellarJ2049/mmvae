#!/bin/bash

OUTPUTDIR="../outputs"

# Dataset generation directory:
root="/data/backed_up/shared/Data/PolyMNIST"
data_DIR="${root}/quadrants_4x_64x64_scl1/Mar19_2025_t0"
DATADIR="${data_DIR}/PolyMNIST"

# Classifier dir:
clf_digit_path="${data_DIR}/clfs_digit/t1_bs512_ep30/DigitClassifier"
clf_quadrant_path="${data_DIR}/clfs_quadrant/t1_bs512_ep10/DigitClassifier"

INCEPDIR=""  # For pt_inception
OBJ="iwae"   # elbo, iwae, dreg
BATCH=128    # default: 128, debug: 32 or 64 if without enough memory
K=1
EPOCHS=150
SEED=2

beta=2.5
priorposterior="Normal" # Normal, Laplace, Diffusion
likelihood="Laplace"
diff_lw=0.0 # 0.1, 1.0
# diff_stop_grad="--diffusion_stop_grad_on_input" # ON: "--diffusion_stop_grad_on_input" or OFF ""
diff_sg="OFF"        # valid values: ON or OFF
if   [ "$diff_sg" = "ON" ]; then
    diff_stop_grad="--diffusion_stop_grad_on_input"
elif [ "$diff_sg" = "OFF" ]; then
    diff_stop_grad=""
else
    echo "Error: diff_sg must be ON or OFF" >&2
    exit 1
fi

dim_shared=32
dim_private=0

indep_scale=0.0
aug_ai_scale=0.0
mvib_scale=0.0

dataset_version="digit_quadrant" # mmvaeplus (MMVAE+ setup), digit_quadrant (New setup)

debug_mini_dataset=false  # true or false
if [ "${debug_mini_dataset}" = true ]; then
    DEBUG_FLAG="--debug_mini_data"
    db="_debug"
    marker="_test"
else
    DEBUG_FLAG=""
    db=""
    marker=""
fi

date=`echo $(date '+%m-%d')`
number=17
gpuid=2
# note="${db}_diff_sg-${diff_sg}_lw${diff_lw}" # usemean and usePost are true by default
note="${db}_allGen_latClf"

EXPERIMENT="PolyMNIST_from_18May25${marker}"
tSNE_save_dir="/data/backed_up/shared/t_SNE/${EXPERIMENT}/${date}_${number}${note}"

CUDA_VISIBLE_DEVICES=${gpuid}  python train_MMVAE_polyMNIST.py --experiment $EXPERIMENT --obj $OBJ --K $K --batch-size $BATCH --epochs $EPOCHS \
      --latent-dim-z $dim_shared --latent-dim-w $dim_private --seed $SEED --beta $beta \
      --datadir $DATADIR  --outputdir $OUTPUTDIR \
      --pretrained_clfs_digit_dir_path $clf_digit_path/trained_clfs_polyMNIST \
      --pretrained_clfs_quadrant_dir_path $clf_quadrant_path/trained_clfs_polyMNIST \
      --dataset_version ${dataset_version} \
      --priorposterior ${priorposterior} \
      --likelihood ${likelihood} \
      --diffusion_loss_weight ${diff_lw} \
      ${diff_stop_grad} \
      --indep_loss_scale $indep_scale --augmented_mi_scale $aug_ai_scale --mvib_loss_scale $mvib_scale \
      --note "${date}_${number}_gpu${gpuid}${note}" \
      --tSNE_save_dir $tSNE_save_dir \
      $DEBUG_FLAG

      # --checkpoint-path $CP_PATH
      # --inception_path "${INCEPDIR}/pt_inception-2015-12-05-6726825d.pth" \
      # --pretrained-clfs-dir-path "${DATADIR}/DigitClassifier/trained_clfs_polyMNIST" \

