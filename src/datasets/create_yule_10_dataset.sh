NUM_REPS=100
CHAIN_LENGTH=17500000
LOG_EVERY=500

# run the yule model to generate trees and alignments and create the corresponding BEAST XML files
# sh "$BEAST_PKG/lphybeast/bin/lphybeast" src/datasets/yule_10.lphy \
#     -o ../../data/lphy/yule-10.xml \
#     -r $NUM_REPS \
#     -l $CHAIN_LENGTH \
#     -le $LOG_EVERY

# run beast to generate the posterior tree samples
cd data/beast
for i in $(seq 37 $(($NUM_REPS - 1))); do
    sh "$BEAST/bin/beast" -overwrite -threads -1 ../lphy/yule-10-$i.xml
done