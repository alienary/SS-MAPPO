# script for train desired roadnets for all models

device=$1
log_dir=$2
all_roadnet=$3
pred_model_dir=$4

file="run_mix.py"

if [ -z "$5" ]; then
  # 未提供，使用默认值
  pred_model_cls="origin"
else
  pred_model_cls="$5"
fi


for roadnet in $all_roadnet
do
    all_veh_roadnet=$(echo $roadnet | sed 's/\(.*\)_[^_]*_[^_]*$/\1/')
    echo $all_veh_roadnet
    python $file $roadnet --model_name "PLPred" --pred_model_dir $pred_model_dir --device $device --log_dir $log_dir --origin_roadnet $all_veh_roadnet --pred_model $pred_model_cls
    python $file $roadnet --model_name "PLTrainPred" --device $device --log_dir $log_dir --origin_roadnet $all_veh_roadnet --pred_model $pred_model_cls
    # python $file $roadnet --model_name "PLNoPred" --device $device --log_dir $log_dir --origin_roadnet $all_veh_roadnet 
    # python $file $all_veh_roadnet --model_name "PLNoPred" --device $device --log_dir $log_dir --mix_traffic
done
