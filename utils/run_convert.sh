roadnet="LF_1x1_13_7_l"
path="../data/"

python roadnet_data_convert.py $roadnet
python flow_data_convert.py $roadnet
