data_names="cifar10.c"
# data_names="cifar10.1 cifar10.2 cifar10.stl lsun svhn cifar10.c"
# data_names="CIFAR10_x32"
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

# model_path=(
#     checkpoints/FED/fed_c100_randaugmixup_2025/checkpoint_249
#     checkpoints/FED/fed_c100_randaugmixup_2026/checkpoint_248
#     checkpoints/FED/fed_c100_randaugmixup_2027/checkpoint_244
# )
# mode=(
#     fed
#     fed
#     fed
# )
# name=(
#     randaugmixup_2025
#     randaugmixup_2026
#     randaugmixup_2027
# )

for data_name in $data_names
do
    for j in ${!model_path[*]}
    do
        python logit_extraction.py --mode ${mode[$j]} --saved_model_path ${model_path[$j]} --data_name $data_name --name ${name[$j]} --data_subname $data_subname --save_path ./logits/ --num_samples 1024 --virtual_num_samples 128
    done
done