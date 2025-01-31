# data_names="cifar10.1 cifar10.2 cifar10.stl lsun svhn cifar10.c"
data_names="CIFAR10_x32"
# data_names="CIFAR100_x32"

data_subname="c10"
# data_subname="c100"

model_path=(
    checkpoints/FED/fed_c10_cinic_2025/checkpoint_235
    checkpoints/FED/fed_c10_randaugmixup_2025/checkpoint_241
    checkpoints/FED/fed_c10_randaugmixup_2026/checkpoint_249
    checkpoints/FED/fed_c10_randaugmixup_2027/checkpoint_231
    checkpoints/FED/fed_c10_simple/checkpoint_239
    checkpoints/FED/fed_c10_simplemixup/checkpoint_996
)
mode=(
    fed
    fed
    fed
    fed
    fed
    fed
)
name=(
    cinic
    randaugmixup_2025
    randaugmixup_2026
    randaugmixup_2027
    simple
    simplemixup
)
for data_name in $data_names
do
    for idx in ${!model_path[*]}
    do
        python logit_extraction.py --mode ${mode[$idx]} --saved_model_path ${model_path[$idx]} --data_name $data_name --name ${name[$idx]} --data_subname $data_subname --save_path ./logits/
    done
done