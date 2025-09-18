# LIDS-Defense-for-Gradient-inversion-in-Federated-Learning
This repository is currently undergoing active refactoring and modularization. 

## Paper 
For the full paper, please see `LIDS_GLOBECOM2025_paper.pdf`

## Train SEER model
To train the required SEER models, please follow the instructions in `seer/README.md`. You should train combinations of the following model parameters and save the trained weights to the `LIDS/weights` folder. Alternatively, you can download the pre-trained SEER model from [Hugging Face (alan314159/LIDS)](https://huggingface.co/alan314159/LIDS) and place it in the `LIDS/weights` folder.

Supported Datasets (DATASET):
- `Cifar10`
- `Cifar100`

Supported Properties (PROPERTY):
- `bright`
- `dark`
- `red`
- `blue`
- `green`
- `hedge`
- `vedge`
- `rand_conv`

Example training command (please refer to `seer/README.md` for details):
```bash
bash train_rel.sh BATCH_SIZE --DATASET DATASET --PROPERTY PROPERTY
```
## LIDS experiments
To run the LIDS experiments, please run the following command:

### Generate LIDS dataset (LIDS-A, LIDS)
```bash
TRAIN=False DATASET=Cifar100 PSNR_THRESHOLD=18.0 bash generate_lids_dataset.sh 
```
### Test SEER with LIDS defense
Before running the end-to-end test, ensure you have generated the LIDS dataset and trained the attack model. Then, from within the `seer/` directory, execute:
```bash
bash test_end2end.sh
```
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


