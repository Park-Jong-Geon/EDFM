data_names="tin"
for data_name in $data_names
do
    seeds="2025 2026 2027"
    for seed in $seeds
    do
    python baseline.py --config config_baseline/kd_${data_name}_frnrelu.yaml --seed $seed --save checkpoints/kd_${data_name}_${seed}
    # python baseline.py --config config/baseline/endd_${data_name}_frnrelu.yaml --seed $seed --save endd_${data_name}_${seed}
    done
done