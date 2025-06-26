# script for train desired roadnets for all models

device=$1
log_dir=$2
all_roadnet=$3
lr=$4
entropy=$5
file="run.py"

for roadnet in $all_roadnet
do
    python $file $roadnet --model_name "PhaseLight" --device $device --log_dir $log_dir --n_exp 10 --lr $lr --entropy $entropy
    python $file $roadnet --model_name "PhaseLightNoencoder" --device $device --log_dir $log_dir --n_exp 10 --lr $lr --entropy $entropy
    python $file $roadnet --model_name "PhaseLight2" --device $device --log_dir $log_dir --n_exp 10 --lr $lr --entropy $entropy
    python $file $roadnet --model_name "PhaseLightNomap" --device $device --log_dir $log_dir --n_exp 10 --lr $lr --entropy $entropy
    python $file $roadnet --model_name "PhaseLight4" --device $device --log_dir $log_dir --n_exp 10 --lr $lr --entropy $entropy
    python $file $roadnet --model_name "FRAP" --device $device --log_dir $log_dir --n_exp 10 --lr $lr --entropy $entropy
    python $file $roadnet --model_name "MLP" --device $device --log_dir $log_dir --n_exp 10 --lr $lr --entropy $entropy
    python run_sotl.py $roadnet --log_dir $log_dir
    python run_mp.py $roadnet --log_dir $log_dir
done
