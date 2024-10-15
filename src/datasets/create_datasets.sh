NUM_REPS=2
CHAIN_LENGTH=100000
LOG_EVERY=1000

# run the yule model to generate trees and alignments and create the corresponding BEAST XML files
sh "$BEAST_PKG/lphybeast/bin/lphybeast" src/datasets/yule.lphy \
    -o ../../data/yule/yule.xml \
    -r $NUM_REPS \
    -l $CHAIN_LENGTH \
    -le $LOG_EVERY

# run beast to generate the posterior tree samples
for i in $(seq 0 $($NUM_REPS - 1)); do
    (
        cd data/beast
        sh "$BEAST/bin/beast" -overwrite ../yule/yule-$i.xml
    )
done