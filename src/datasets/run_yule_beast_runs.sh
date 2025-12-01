# run the BEAST 2 xmls

NUM_REPS=50

mkdir data/mcmc
cd data/mcmc

unset BEAST

for n in 50; do
    for i in $(seq 1 $(($NUM_REPS - 1))); do
        if [ "$HPC" == "true" ]; then
            echo "Submitting job for yule-$n-$i"
            sbatch --time=24:00:00 --mem=4G --cpus-per-task=16 --ntasks=1 --hint=multithread -o out_err/out-yule-$n-$i -e out_err/err-yule-$n-$i "$BEAST/bin/beast" -overwrite -threads -1 ../lphy/yule-$n-$i.xml
            sleep 1
        else
            echo "Running BEAST for yule-$n-$i"
            phylorun ../lphy/yule-$n-$i.xml -overwrite -threads -1
        fi
    done
done
