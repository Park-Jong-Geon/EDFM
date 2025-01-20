data_names="CIFAR100_x32 lsun svhn"
for data_name in $data_names
do
    python logit_extraction.py --mode endd --saved_model_path checkpoints/endd_c100_2028/checkpoint_981 --data_name $data_name --name 2028
    # python logit_extraction.py --mode endd --saved_model_path checkpoints/endd_c10_2027/checkpoint_986 --data_name $data_name --name 2027
    # python logit_extraction.py --mode endd --saved_model_path checkpoints/endd_c10_2028/checkpoint_999 --data_name $data_name --name 2028
    # python logit_extraction.py --mode endd --saved_model_path checkpoints/endd_c10_2029/checkpoint_999 --data_name $data_name --name 2029

    # python logit_extraction.py --mode kd --saved_model_path checkpoints/kd_c10_2025/checkpoint_985 --data_name $data_name --name 2025
    # python logit_extraction.py --mode kd --saved_model_path checkpoints/kd_c10_2026/checkpoint_994 --data_name $data_name --name 2026
    # python logit_extraction.py --mode kd --saved_model_path checkpoints/kd_c10_2027/checkpoint_993 --data_name $data_name --name 2027
done