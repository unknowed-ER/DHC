export TF_CPP_MIN_LOG_LEVEL=2
Gpus=0
Droot=cached   # default cached
Mname=SKT_KG
Cfg=default.yml

python train.py \
    --cfg ymls/$Cfg \
    --cache_dir $Droot \
    --gpus $Gpus \
    --checkpoint_dir checkpoints/pchat/SKT_KG/2021-12-28-21-34-07_rerun_GS \
    $Mname




