# LIDS-Defense-for-Gradient-inversion-in-Federated-Learning
This repository is currently undergoing active refactoring and modularization. We target an initial runnable release by 15 May 2025, with the project roadmap and outstanding tasks to be continuously tracked and updated.

## Train SEER model
To train the SEER model of different datasets and properties, please run the following command:

DATASET: `Cifar10`, `Cifar100`
PROPERTY: `bright`, `dark`, `red`, `blue`, `green`, `hedge`, `vedge`, `rand_conv`

```bash
bash train_rel.sh BATCH_SIZE --dataset DATASET --prop PROPERTY
```
Or you can download the pre-trained SEER model from [huggingface](https://huggingface.co/alan314159/LIDS) and put it in the `LIDS/weights` folder.

## LIDS experiments
To run the LIDS experiments, please run the following command:

### LIDS without augmentation (LIDS-A)

### LIDS (LIDS)


### Reconstruction Label Accuracy (RLA)
To run the RLA experiments in TABLE 2, please download the pre-trained dinov2 model from [huggingface](https://huggingface.co/alan314159/LIDS) and put it in the `LIDS/weights` folder, then run the following command:

```bash
bash test_label_acc.sh
```
the results will be saved in the `LIDS/logs/RLA_results.txt` file.

### Global Model Test Accuracy

To run the global model test accuracy experiments in TABLE 3, please run the following command:

```bash
python3 train_resnet.py --dataset Cifar10
python3 train_resnet.py --dataset Cifar100
python3 train_resnet.py --dataset Cifar10 --lids
python3 train_resnet.py --dataset Cifar100 --lids
```

the results will be saved in the `LIDS/logs` folder.