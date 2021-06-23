#!/bin/bash


MYGIT=/mnt/nvme/zcq/git
REPO_PATH=/home/fulan/shadow_removal
# DATA_PATH=${MYGIT}/shadow_removal/data/SRD
# datasetmode=srd

DATA_PATH=/home/fulan/ShadowRemoval/data
datasetmode=expo_param


batchs=4
n=5
ks=3
rks=3
version='fixed5-1-loss'

lr_policy=lambda
lr_decay_iters=50
optimizer=adam
shadow_loss=10.0

tv_loss=0
grad_loss=0.0
pgrad_loss=0.1

gpus=0


lr=0.0001
loadSize=256
fineSize=256
L1=10
model=Fusion
load_dir=None
checkpoint=${REPO_PATH}/log
dataroot=${DATA_PATH}
NAME="M${model}_${datasetmode}_b${batchs}_lr${lr}_L1${L1}_n${n}_ks${ks}_v${version}_${optimizer}_${lr_policy}_${shadow_loss}_TV${tv_loss}G${grad_loss}PG${pgrad_loss}"
OTHER="--save_epoch_freq 100 --niter 50 --niter_decay 350"


trainmask=${dataroot}'/train_NOTUSE'
CMD="python -u ./OE_train.py --loadSize ${loadSize} \
    --randomSize
    --name ${NAME} \
    --dataroot  ${dataroot}\
    --checkpoints_dir ${checkpoint} \
    --fineSize $fineSize --model $model \
    --batch_size $batchs \
    --randomSize --keep_ratio --phase train_  --gpu_ids ${gpus} --lr ${lr} \
    --lambda_L1 ${L1} --num_threads 16 \
    --dataset_mode $datasetmode\
    --mask_train $trainmask --optimizer ${optimizer} \
    --n ${n} --ks ${ks} --lr_policy ${lr_policy} --lr_decay_iters ${lr_decay_iters} \
    --shadow_loss ${shadow_loss} --rks ${rks} --tv_loss ${tv_loss} --grad_loss ${grad_loss} --pgrad_loss ${pgrad_loss} \
    --load_dir ${load_dir} \
    $OTHER
"
echo $CMD
eval $CMD # >> ${checkpoint}/${NAME}.log 2>&1 &

