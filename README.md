# HyperSynergyV4

以 HyperSynergyV2 的当前完整源码为基线，加入 Stage2 OT Flow Matching 辅助先验，以及用户批准的早停：每200个训练step检查验证LOSS，连续5次未严格降低则停止。原有损失、编码器和预测路径保留。

**这是代码实现快照，尚未完成真实数据集成验证，也没有性能提升结论。** 已完成的单元测试和待补测项目见 [实现说明](docs/OTFM.md) 与 [审计记录](docs/OTFM_AUDIT.md)。用户在明确知悉待补测状态后要求上传本版本。

OT-FM 默认关闭；V4实验配置为 configs/hyperV4_otfm.json（enabled=true，weight=0.02；仅初始值）。本次未进行正式训练，仓库不包含运行日志、checkpoint或账号凭据。

保留原 HyperSynergy Git 历史。下面为原项目说明。

---


# HyperSynergy

Source code and dataset for TPAMI 2023 [paper](https://ieeexplore.ieee.org/abstract/document/10050784/): "Few-Shot Drug Synergy Prediction with a Prior-Guided Hypernetwork Architecture"

## Requirements

Python == 3.6

torch == 1.7.0+cu110

torch-geometric == 1.6.1

torch-scatter == 2.0.5

torch-sparse == 0.6.8

torch-cluster == 1.5.8                                     

rdkit == 2019.09.3.0

pytorchtools == 0.0.2


## Usage (Step by step runing)

### 0. Create Sample Feature Dictionaries （Required when running on other datasets）

If you would like to run the code of the HyperSynergy on other datasets, please first obtain the drugs' sdf file from [PubChem](https://pubchem.ncbi.nlm.nih.gov/) through the drugs' Pubchem cid, and gene expression profile (900 genes, **gene_expression_examples.csv**) of cell lines from [CCLE](https://depmap.org/portal/download/all/) or [GDSC](https://www.cancerrxgene.org/downloads/bulk_download), and then create three .p files: cell_feature_900.p, codes_cell.p, and drug_feature_cell.p by running feature_generation.py and feature_direction.py

### 1. Representation Learning

If you would like to get a pretrained feature embedding model, please run representation_learning.py

### 2. Meta-training 

If you would like to learn the weights and biases of meta-generative network module of the HyperSynergy, please run train_few_shot.py

### 3. Meta-test

If you would like to test the power of HyperSynergy on meta-test cell lines, please run test_few_shot.py

## Citation

If you find our work is useful for your research, please consider citing our paper:

@article{zhang2023few,<br>
  title={Few-Shot Drug Synergy Prediction With a Prior-Guided Hypernetwork Architecture},<br>
  author={Zhang, Qingqing and Zhang, Shaowu and Feng, Yuehua and Shi, Jianyu},<br>
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},<br>
  year={2023},<br>
  publisher={IEEE}
}
