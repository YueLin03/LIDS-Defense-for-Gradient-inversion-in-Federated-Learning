dataset="Cifar10"
colors=("blue" "bright" "dark" "green" "red" "rand_conv" "hedge" "vedge")
thresh=("17.0" "18.0" "19.0" "20.0" "21.0" "22.0")
for color in "${colors[@]}"; do
    for t in "${thresh[@]}"; do
        python3 label_acc.py --thresh "$t" --attack_prop "$color" --dataset "$dataset" --device "cuda:0"
    done
done

dataset="Cifar100"
colors=("blue" "bright" "dark" "green" "red" "rand_conv" "hedge" "vedge")
thresh=("17.0" "18.0" "19.0" "20.0" "21.0" "22.0")
for color in "${colors[@]}"; do
    for t in "${thresh[@]}"; do
        python3 label_acc.py --thresh "$t" --attack_prop "$color" --dataset "$dataset" --device "cuda:0"
    done
done