ips=(
    35.186.67.37 
    107.167.166.81 
    108.59.90.87 
    35.186.61.106 
    35.186.77.80 
    107.167.174.141
)

accounts=(
    panyaang99
    panyaang99
    panyaang99
    panyaang99
    giung
    giung
)

ckpts=(
    c100_randaugmixup_improvedkd
    c100_randaugmixup_improvedkd_2026
    c100_randaugmixup_improvedkd_2027
    fed_c100_2026
    fed_c10_simple
    fed_c10_simplemixup
)


for idx in ${!ips[*]}
do
    scp -i ~/.ssh/tpu_openssh -r  ${accounts[$idx]}@${ips[$idx]}:~/dbn/checkpoints/${ckpts[$idx]}  ~/dbn/checkpoints
done