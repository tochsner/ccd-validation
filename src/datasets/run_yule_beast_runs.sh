# run the yule models to generate trees and alignments and create the corresponding BEAST XML files

export BEAST="/nesi/nobackup/nesi00390/tobia/beast/beast"

# run beast to generate the posterior tree samples
cd data/mcmc_relicas

for n in 10 50 100 200 400; do
    for i in $(seq 1 $(($NUM_REPS - 1))); do
        echo "Submitting job for yule-$n-$i"
        sbatch --time=24:00:00 --mem=4G --cpus-per-task=16 --ntasks=1 --hint=multithread -o out_err/out-yule-$n-$i -e out_err/err-yule-$n-$i "$BEAST/bin/beast" -overwrite -threads -1 ../lphy/yule-$n-$i.xml
        sleep 1
        exit 1
    done
done
