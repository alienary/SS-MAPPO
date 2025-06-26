# scripts run test for one roadnet for all models
roadnet=$1
device=$2
log_dir=$3
n_exp=$4
model_dir=$5
file="run.py"

model_name="PhaseLight PhaseLightNomap PhaseLightNoencoder PhaseLight4 PhaseLight2 FRAP MLP"
for model in $model_name
do
cur_model_dir="$model_dir/$model"
# python $file $roadnet --model_name $model --device $device --log_dir $log_dir --n_exp $n_exp --test --model_dir $cur_model_dir >/dev/null 2>&1 &
python $file $roadnet --model_name $model --device $device --log_dir $log_dir --n_exp $n_exp --test --model_dir $cur_model_dir 
done
wait
echo "Ok"
