export TF_CPP_MIN_LOG_LEVEL=2

# for ckptfname in \
# './checkpoints/pchat/PIPM_ori/20211207084557_rerun/ckpt-270000' \
# './checkpoints/pchat/PIPM_ori/20211207084557_rerun/best_checkpoints/ckpt-1th-best' \
# ;
# do
# echo $ckptfname
# python inference.iclr.py \
#     --cfg ymls/default.yml \
#     --gpus 0 \
#     --test_mode wow \
#     --test_ckpt $ckptfname \
#     PIPM
# done

# for ckptfname in \
# './checkpoints/pchat/SequentialKnowledgeTransformer/20211220152202_rerun/ckpt-69000' \
# ;
# do
# echo $ckptfname
# python inference.iclr.py \
#     --cfg ymls/default.yml \
#     --gpus 0 \
#     --test_mode wow \
#     --test_ckpt $ckptfname \
#     SequentialKnowledgeTransformer
# done

for ckptfname in \
'./checkpoints/pchat/SKT_KG/20211220111207_rerun/ckpt-96000' \
'./checkpoints/pchat/SKT_KG/20211220111207_rerun/best_checkpoints/ckpt-2th-best' \
'./checkpoints/pchat/SKT_KG/20211220111207_rerun/best_checkpoints/ckpt-1th-best' \
;
do
echo $ckptfname
python inference.iclr.kg.py \
    --cfg ymls/default.yml \
    --gpus 0 \
    --test_mode wow \
    --test_ckpt $ckptfname \
    SKT_KG
done

# python inference.iclr.kg.py --cfg ymls/default.yml --gpus 0 --test_mode wow --test_ckpt 20211220111207_rerun/best_checkpoints/ckpt-1th-best SKT_KG