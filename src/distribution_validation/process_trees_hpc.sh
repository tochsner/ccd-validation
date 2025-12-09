# run the BEAST 2 xmls

NUM_REPS=100

mkdir -p data
mkdir -p data/processed

for n in 50; do
    for i in $(seq 1 99); do
        if [ "$HPC" = "true" ]; then
            echo "Submitting job for yule-$n-$i"
            sbatch --time=24:00:00 --mem-per-cpu=4G --cpus-per-task=16 -o out_err/out-yule-$n-$i -e out_err/err-yule-$n-$i --wrap="java -jar src/jars/ProcessTrees.jar yule-$n-$i.trees data/mcmc data/processed"
            sleep 1
        else
            java -jar src/jars/ProcessTrees.jar yule-$n-$i.trees data/mcmc data/processed
        fi
    done
done
