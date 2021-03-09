# Auto-exposure fusion for single-image shadow removal
We propose a new method for effective shadow removal by regarding it as an exposure fusion problem. Please refer to the paper for details: https://arxiv.org/abs/2103.01255

![Framework](./images/framework.png)

## Dataset

- ISTD https://github.com/DeepInsight-PCALab/ST-CGAN
- ISTD+ https://github.com/cvlab-stonybrook/SID
- SRD

## Train

Modify the corresponding path in file `OE_train.sh` and run the following script

```shell
sh OE_train.sh
```


## Test

In order to test the performance of a trained model, you need to make sure that the hyper parameters in file `OE_eval.sh` match the ones in `OE_train.sh` and run the following script

```shell
sh OE_test.sh
```

The results reported in the paper are calculated by the `matlab` script used in other SOTA, please see https://github.com/cvlab-stonybrook/SID/issues/1 for details. Our evaluation code will print the metrics calculated by `python` code and save the result images which will be used by the `matlab` script.

## Results

- Comparsion with SOTA, see paper for details.

![Framework](./images/vis_compare.png)


- Penumbra comparsion between ours and SP+M Net

![Framework](./images/edge_comparsion.png)

- Testing result

The testing results on dataset ISTD+ is:
https://drive.google.com/file/d/1tX4wDW3gqQcvN1Qo2QOZb0INdwXLymgS/view?usp=sharing

The testing results on dataset SRD is:
https://drive.google.com/file/d/1deXG9s0RPJut_E-vdTRwhi7QWnHjwoSh/view?usp=sharing
**More details are coming soon**

## Bibtex

```
@inproceedings{fu2021auto,
      title={Auto-exposure Fusion for Single-image Shadow Removal}, 
      author={Lan Fu and Changqing Zhou and Qing Guo and Felix Juefei-Xu and Hongkai Yu and Wei Feng and Yang Liu and Song Wang},
      year={2021},
      booktitle={accepted to CVPR}
}
```
