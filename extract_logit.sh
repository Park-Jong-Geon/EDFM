data_names="cifar10.1 cifar10.2 cifar10.stl lsun svhn cifar10.c"
data_subname="c10"
model_path=(
    checkpoints/done/endd_c10_cinic/checkpoint_998
    checkpoints/done/endd_c10_randaugmixup/checkpoint_984
    checkpoints/done/endd_c10_simple/checkpoint_950
    checkpoints/done/kd_c10_randaugmixup/checkpoint_987
    checkpoints/done/kd_c10_simple/checkpoint_952
)
mode=(
    endd
    endd
    endd
    kd
    kd
)
name=(
    cinic
    randaugmixup
    simple
    randaugmixup
    simple
)
for data_name in $data_names
do
    for idx in ${!model_path[*]}
    do
        python logit_extraction.py --mode ${mode[$idx]} --saved_model_path ${model_path[$idx]} --data_name $data_name --name ${name[$idx]} --data_subname c10 --save_path ./logits/ood/
    done
done