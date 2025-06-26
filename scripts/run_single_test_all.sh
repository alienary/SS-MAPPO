# scripts run test for all train roadnet and test roadnet


# export CUDA_VISIBLE_DEVICES=2

# roadnet=$1
device=cuda:0
log_dir=$1
model_dir=$2
test_roadnet="syn1_1x1_1h syn2_1x1_1h syn3_1x1_1h syn4_1x1_1h syn5_1x1_1h syn6_1x1_1h LF_1x1_53_8_l hangzhou_1x1_bc-tyc_18041608_1h"
train_roadnet="syn1_1x1_1h syn2_1x1_1h syn3_1x1_1h syn4_1x1_1h syn5_1x1_1h syn6_1x1_1h LF_1x1_53_8_l hangzhou_1x1_bc-tyc_18041608_1h"

n_exp=10
for r in $train_roadnet
do
new_log_dir="$log_dir/train_$r"
new_model_dir="$model_dir/$r"
for roadnet in $test_roadnet
do
    # bash run_single_test.sh $roadnet $device $new_log_dir $n_exp $new_model_dir >/dev/null 2>&1 
    bash run_single_test.sh $roadnet $device $new_log_dir $n_exp $new_model_dir 
done
python script/stat_result_single.py $new_log_dir --test
echo $new_log_dir
done
