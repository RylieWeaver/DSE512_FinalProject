plot_root=/mnt/DGX01/Personal/r9w/Checkpoints/Microbial/genome_regression_seed42

for file in "$plot_root"/scenario*/{mlp/plots,transformer/plots,plots}/*.{png,pdf}; do
    [ -f "$file" ] || continue
    name=${file#"$plot_root"/}
    cp -v -- "$file" "./plots/${name//\//_}"
done
