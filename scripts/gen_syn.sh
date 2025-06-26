$(python utils/random_trips.py data/syn1_1x1_1h 0.7 4 --random_seed 42 --min_prob 0.7 --max_prob 0.8)
$(python utils/random_trips.py data/syn2_1x1_1h 0.7 4 --random_seed 43)
# $(python utils/random_trips.py data/syn2_1x1_1h 0.65 4 --random_seed 43 --min_prob 0.8 --max_prob 0.9)
$(python utils/random_trips.py data/syn3_1x1_1h 0.68 4 --random_seed 44)
$(python utils/random_trips.py data/syn4_1x1_1h 0.65 4 --random_seed 45)
$(python utils/random_trips.py data/syn5_1x1_1h 0.65 4 --random_seed 46)
$(python utils/random_trips.py data/syn6_1x1_1h 0.65 4 --random_seed 47)
echo "syn1_1x1_1h"
python utils/remove_loops.py data/syn1_1x1_1h
echo "syn2_1x1_1h"
python utils/remove_loops.py data/syn2_1x1_1h
echo "syn3_1x1_1h"
python utils/remove_loops.py data/syn3_1x1_1h
echo "syn4_1x1_1h"
python utils/remove_loops.py data/syn4_1x1_1h
echo "syn5_1x1_1h"
python utils/remove_loops.py data/syn5_1x1_1h
echo "syn6_1x1_1h"
python utils/remove_loops.py data/syn6_1x1_1h
echo "LF_1x1_53_8_l"
python utils/remove_loops.py data/LF_1x1_53_8_l
echo "hangzhou_1x1_bc-tyc_18041608_1h"
python utils/remove_loops.py data/hangzhou_1x1_bc-tyc_18041608_1h
# echo "LF_1x1_13_8_l"
# python utils/remove_loops.py data/LF_1x1_13_8_l

