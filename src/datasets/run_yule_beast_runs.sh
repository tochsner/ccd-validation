# run the BEAST 2 xmls

NUM_REPS=100

mkdir -p data
mkdir -p data/mcmc

cd data/mcmc

for n in 50; do
    for i in $(seq 1 $(($NUM_REPS - 1))); do
        if [ "$HPC" = "true" ]; then
            echo "Submitting job for yule-$n-$i"
            sbatch --time=24:00:00 --mem-per-cpu=4G --cpus-per-task=16 -o out_err/out-yule-$n-$i -e out_err/err-yule-$n-$i --wrap="../../../../beast/bin/beast -overwrite ../lphy/yule-$n-$i.xml"
            sleep 1
        else
            echo "Running BEAST for yule-$n-$i"
            unset BEAST
            phylorun ../lphy/yule-$n-$i.xml -overwrite -threads -1
        fi
    done
done
