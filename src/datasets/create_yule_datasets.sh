# run the yule models to generate trees and alignments and create the corresponding BEAST XML files

NUM_REPS=100
CHAIN_LENGTH=35000000
LOG_EVERY=1000

export BEAST_PKG="/home/toch456/.beast/2.7"
export LPHY_LIB="/nesi/nobackup/nesi00390/tobia/beast/lphy/lib"
export BEAST="/nesi/nobackup/nesi00390/tobia/beast/beast"

for n in 10 50 100 200 400; do
    sh "$BEAST_PKG/lphybeast/bin/lphybeast" ../src/datasets/yule-$n.lphy \§
        -o ../../data/lphy/yule-$n.xml \
        -r $NUM_REPS \
        -l $CHAIN_LENGTH \
        -le $LOG_EVERY
done
