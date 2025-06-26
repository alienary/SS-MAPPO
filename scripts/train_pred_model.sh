device=$1
log_dir=$2

file="train_pred_model.py"


if [ -z "$3" ]; then
  # 未提供，使用默认值
  model="origin"
else
  model="$3"
fi
nohup python $file data/inter_state --device $device --log_dir $log_dir --seed 42 --model $model >/dev/null 2>&1 &
nohup python $file data/inter_state --device $device --log_dir $log_dir --seed 47 --model $model >/dev/null 2>&1 &
nohup python $file data/inter_state --device $device --log_dir $log_dir --seed 52 --model $model >/dev/null 2>&1 &
nohup python $file data/inter_state --device $device --log_dir $log_dir --seed 57 --model $model >/dev/null 2>&1 &
nohup python $file data/inter_state --device $device --log_dir $log_dir --seed 62 --model $model >/dev/null 2>&1 &
