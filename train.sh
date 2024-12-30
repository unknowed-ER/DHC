export TF_CPP_MIN_LOG_LEVEL=2
Gpus=0
Droot=cached
Mname=SKT_KG
Cfg=default.yml

python train.py --cfg ymls/$Cfg --cache_dir $Droot --gpus $Gpus $Mname

# python train.py --cfg ymls/default.yml --cache_dir cached --gpus 0 SKT_KG