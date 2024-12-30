export TF_CPP_MIN_LOG_LEVEL=2
Gpus=0
Droot=cached   # default cached
Mname=SequentialKnowledgeTransformer
Cfg=default.yml

python train.iclr.py \
    --cfg ymls/$Cfg \
    --cache_dir $Droot \
    --gpus $Gpus \
    $Mname
