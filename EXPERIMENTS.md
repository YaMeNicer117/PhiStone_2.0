# PhiStone 实验设计与结果说明

## 1. 实验概述与结果资源

本文档介绍 PhiStone 竞赛项目的实验目的、数据输入、生成与筛选方法、模型对照、评价指标及结果归档位置。实验围绕抗结核先导分子优化、活性预测模型的数据策略与消融分析，以及口袋条件分子生成的基线比较展开。

本文中的项目路径均相对于 `PhiStone_2.0_UP/` 根目录。数据准备流程见 [DATA_PREPARATION.md](DATA_PREPARATION.md)，模型训练与交互式推理入口见 [README.md](README.md)。

### 1.1 实验组成

| 实验类别 | 实验目的 | 主要方法 | 结果目录 |
| --- | --- | --- | --- |
| 抗结核小分子生成与筛选 | 基于白皮杉醇先导结构开展片段修饰与候选分子进一步优化，筛选待合成化合物。 | 三种生成策略、SA/QED 过滤、分子去重、SMINA 位置校正、亲和力与活性预测、人工复核。 | `results/generated_molecules/` |
| PhiSGATv2 数据策略与全局均值消融 | 比较训练数据来源与虚拟节点设置对外部测试数据预测误差的影响。 | ALLIN 主模型、ALLIN 全局均值消融组与 MIC-only 数据策略对照组。 | `results/PhiGATv2_ablation/` |
| 分子生成模式与基线模型比较 | 比较相同先导片段条件下的生成结构合理性、几何与构象、结构差异和预测活性。 | PhiStone 两种生成模式与七个外部基线模型，共 9 个实验组。 | `results/Comparative Evaluation/` |

三类实验的具体设计分别见第 2、3、4 节。

### 1.2 全部实验结果获取

本项目的全部实验结果可通过以下资源获取：

| 资源名称 | 资源用途 | 下载链接 | 提取码 |
| --- | --- | --- | --- |
| `results.tar` | 项目全部实验结果。 | [百度网盘](https://pan.baidu.com/s/1LYNDtK55khfq9CDpkfk0vw?pwd=9x4r) | `9x4r` |

各类实验的项目内归档路径见上表及对应章节。具体评估数值以结果文件中的记录为准。

**补充说明：** 生成分子部分来源于PhiStone_1.0按策略①生成。生成结果已纳入 `results/generated_molecules/`。PhiStone_1.0 采用 BioLiP 与 PDBbind 复合物数据，以 ChemBERTa 嵌入为初始输入训练小型编码器；未加入 HAC 与环数预测头，损失计算与 PhiStone_2.0 略有差异，未进行抗结核活性训练，其余主体训练流程相同。PhiStone_1.0具体代码架构已上传到GitHub仓库，详细介绍见 [PhiStone_1.0 项目仓库](https://github.com/YaMeNicer117/PhiStone_1.0) 中的 README 对该版本的简要介绍。

## 2. 抗结核小分子生成与逐级筛选实验

### 2.1 实验目的与输入

本实验以复合物中的配体白皮杉醇为初始先导，分别修饰乙烯连接部分与羟基部分，并使用生成候选 `0099.sdf` 进一步优化。三种策略的生成结果统一进入后续整理、预测与人工复核流程。

| 输入或入口 | 项目相对路径 |
| --- | --- |
| 复合物 PDB | `Datas/work_datas/input_pdb/PIT_8J0S.pdb` |
| 用于进一步优化的候选分子 | `Datas/work_datas/input_pdb/0099.sdf` |
| 交互式生成入口 | `PhiStone_generate_SE3TD.ipynb` |

### 2.2 三种分子生成策略

在生成 Notebook 中分别设置以下三种策略，均采用 `activity_modify` 模式。

| 策略 | 输入及先导片段设置 | 生成模式 |
| --- | --- | --- |
| ① 乙烯连接部分修饰 | 输入 `PIT_8J0S.pdb`，选择配体白皮杉醇，保留除乙烯键对应片段以外的所有片段作为先导片段。 | `activity_modify` |
| ② 羟基部分修饰 | 输入 `PIT_8J0S.pdb`，选择配体白皮杉醇，保留除羟基片段以外的所有片段作为先导片段。 | `activity_modify` |
| ③ 候选分子进一步优化 | 使用前两种策略所得候选分子中的 `0099.sdf`，完整输入并保留全部片段作为先导片段，继续生成分子。 | `activity_modify` |

策略①与策略②分别考察不同片段保留条件下的生成结果，策略③在前两种策略所得候选分子的基础上进一步优化。

### 2.3 生成结果整理与多层筛选

```text
策略①、策略② → 候选 0099.sdf → 策略③进一步生成

汇集三种策略的全部生成分子
  ↓ reorganize.py
汇总整理生成结果
  ↓ batch_filter_smina.py
SA、QED 过滤、分子去重及 SMINA 位置校正
  ↓
final_docked_unique.sdf
  ↓ 外部模型 PbcNet2.0
亲和力预测与筛选
  ↓
top_affinity_molecules.sdf
  ↓ PhiSGATv2_predict_x.py
抗结核活性预测
  ↓
PhiSGATv2_prediction_results
  ↓ 人工复核与筛选
待合成化合物集
```

| 顺序 | 处理阶段 | 处理入口或方法 | 主要产出 |
| --- | --- | --- | --- |
| 1 | 生成结果汇总 | 使用 `results/generated_molecules/oringin_generated_molecules/reorganize.py` 整理三种策略产生的全部分子。 | 汇总后的生成分子集。 |
| 2 | 理化与合成可及性筛选、位置校正 | 使用 `results/generated_molecules/batch_filter_smina.py` 进行 SA、QED 过滤、分子去重和 SMINA 位置校正。 | `final_docked_unique.sdf` |
| 3 | 外部亲和力预测 | 将筛选结果输入外部模型 PbcNet2.0，预测亲和力并进一步筛选。 | `top_affinity_molecules.sdf` |
| 4 | 抗结核活性预测 | 使用 `PhiSGATv2/PhiSGATv2_predict_x.py` 预测候选分子的抗结核活性。 | `PhiSGATv2_prediction_results` |
| 5 | 人工复核 | 综合预测与评估结果进行人工复核和筛选。 | 待合成化合物集。 |

SA、QED 和 SMINA 处理为生成完成后的独立实验评估步骤，由结果处理脚本执行。

### 2.4 结果保存位置

全部生成分子、各项评估值及各层筛选得到的分子集统一保存在：

```text
results/generated_molecules/
```

最终筛选得到的 **12 个候选分子**的原始生成文件及汇总 XLSX 表格同时归档于 `results/Candidate_molecules/`。

## 3. PhiSGATv2 数据策略与全局均值消融实验

### 3.1 实验目的与数据来源

本实验通过三组 PhiSGATv2 训练结果，比较训练数据输入策略及全局均值消融设置对外部测试数据预测误差的影响。

| 数据策略 | 训练数据源表格 | 数据处理特点 |
| --- | --- | --- |
| ALLIN | `Molecule extraction/Data_Process/ALLIN/activity_mic_level_mean.xlsx` | 综合利用 MIC、IC50 和抑制率信息，按项目规则标准化并汇总分子活性。 |
| MIC-only | `Molecule extraction/Data_Process/MICONLY/human_mtb_unique_smiles_activity_mean.xlsx` | 仅使用 MIC 数据，经过对应的标准化、分子去重与活性汇总。 |

两套数据的清洗、标准化和汇总方式不同，具体流程见 [DATA_PREPARATION.md](DATA_PREPARATION.md) 第 2.3、2.4 节。上表列出训练数据源表格，模型训练使用相应预处理后的图数据。

### 3.2 三组训练设置与权重归档

| 实验组 | 训练数据源 | 虚拟节点设置 | 权重及训练产物目录 |
| --- | --- | --- | --- |
| ALLIN 主模型 | `activity_mic_level_mean.xlsx` | 开启：`USE_VIRTUAL_NODE = True` | `PhiSGATv2/training_artifacts-allinbest/` |
| ALLIN 全局均值消融组 | `activity_mic_level_mean.xlsx` | 关闭：`USE_VIRTUAL_NODE = False` | `PhiSGATv2/training_artifacts-globelmean_ablation experiment/` |
| MIC-only 数据策略对照组 | `human_mtb_unique_smiles_activity_mean.xlsx` | 待补充。 | `PhiSGATv2/training_artifacts_miconly/` |

ALLIN 主模型与全局均值消融组使用同一数据源，通过开启和关闭虚拟节点进行对比；MIC-only 组用于训练数据策略对照，其虚拟节点设置尚未记录。

实际测试使用的主模型权重保存在 `PhiSGATv2/training_artifacts-allinbest/`，另外两组权重用于上述消融与对照实验。上述目录为本次实验归档位置，其中 `training_artifacts-globelmean_ablation experiment` 保留项目实际目录名的拼写与空格。

### 3.3 外部测试数据上的预测误差分析

使用 `PhiSGATv2/PhiSGATv2_activity_curve_evaluate.py` 分别评估三组权重。外部真实测试数据集为：

```text
Datas/work_datas/test_unknownactivity/testdatas_noh.xlsx
```

三组权重均在该外部数据集上进行预测误差分析，用于比较不同训练数据策略及全局均值消融设置下的模型表现。

评估结果统一保存于：

```text
results/PhiGATv2_ablation/
```

具体评估值以该目录中的结果记录为准。

## 4. 分子生成模式与基线模型对比实验

### 4.1 实验目的与共同输入

本实验使用 PhiStone 的 `activity_modify`、`survival_linker` 两种模式，以及 **Delete、DiffGui、DiffLinker、DiffSBDD、PMDM、PocketXMol、FLOWR** 七个基线模型，按照第 2.2 节的白皮杉醇策略①开展口袋条件分子生成。

共同片段条件为：移除连接两侧芳香环的乙烯基，保留两个二羟基苯片段，共 **16 个重原子**，生成新的连接子及各模型设置允许的新增部分。

| 输入或评估入口 | 项目相对路径 |
| --- | --- |
| 输入复合物 | `Datas/work_datas/input_pdb/PIT_8J0S.pdb` |
| 白皮杉醇参考配体 | `results/Comparative Evaluation/ref_ligand/reference_ligand.sdf` |
| 评估脚本 | `results/Comparative Evaluation/evaluate_generated_molecules.py` |

PhiStone 两种模式各生成 100 个分子，七个外部模型采用第 4.2、4.3 节所列的采样与筛选流程。共设置 9 个实验组，评估脚本分别分析各组实际获得的分子集。

### 4.2 外部模型的条件生成设置

| 模型 | 条件生成方式与主要设置 |
| --- | --- |
| Delete | 使用 `linker_val_37.pt`，束宽 500、最大生成步数 50，基于 15 组预设片段空间取向和随机种子生成。 |
| DiffGui | 使用 `trained.pt` 及键预测器 `bond_trained.pt`，采用 `frag_cond`，批量大小 8；按均值约 24.92、标准差约 5.52 的分布抽样总原子数，并要求其大于固定片段原子数。 |
| DiffLinker | 使用口袋条件模型，均匀抽样生成 2–5 个原子的连接子。 |
| DiffSBDD | 使用 `crossdocked_fullatom_cond.ckpt` 进行子结构补全，以口袋中心为采样中心，每轮新增 2–5 个原子。 |
| PocketXMol | 使用预训练模型的原生 `maskfill` 模式，固定保留核心并指定允许连接的位点。 |
| PMDM | 使用预训练模型的 `linker_sample` 固定片段条件采样，采用 `generalized` 方式及检查点规定的扩散步数。 |
| FLOWR | 使用预训练模型的子结构 `inpainting`，设置 100 个积分步、`uniform-sample` 离散采样策略及零附加坐标噪声。 |

### 4.3 采样分组与生成后筛选

**Delete、DiffGui、DiffLinker、DiffSBDD** 使用受体口袋 10 Å 范围内的蛋白结构作为条件，固定上述核心，连接位点由模型决定。生成后按照芳香环与羟基保留、分子连通和结构唯一性进行筛选，每个模型保留 100 个候选分子。

**PocketXMol、PMDM、FLOWR** 直接使用预训练模型，不重新训练。每个模型请求 100 次采样，分组与新重原子预算如下：

| 采样组 | 请求采样次数 | 新重原子预算 |
| --- | --- | --- |
| 连接子对照组 | 34 | `2/3/4` |
| 单点增加组 | 33 | `3/5/7` |
| 双点增加组 | 33 | `4/7/10` |

采用原始姿态与两种预设的小幅刚体扰动姿态，记录模型、分组、随机种子和预算。连接子与新增取代基在一次完整采样中共同生成。

### 4.4 评价指标

| 指标 | 定义与统计依据 |
| --- | --- |
| 分子合理性异常比例 | 无法解析、RDKit 结构检查失败或不是单一连通结构的记录占比。 |
| PoseBusters 几何与构象通过比例 | 键长、键角、芳香环平面性、非芳香环构象及双键平面性五项均通过的记录占比。 |
| 与参考分子平均差异 | 基于 Morgan 指纹的 `1 - Tanimoto`，衡量生成分子与白皮杉醇的平均结构差异。 |
| 集合内部平均差异 | 各组分子两两之间的平均指纹差异，用于比较组内结构多样性。 |
| 预测活性达标数量 | 从已有 PhiSGATv2 预测结果中统计预测活性严格大于 `0.45` 的记录数。 |

### 4.5 结果保存位置

生成分子及运行结果保存在 `results/Comparative Evaluation/`，评估汇总表、逐分子结果及活性达标数量图默认输出到其 `evaluation_results/` 子目录。
|

实验结果可通过第 1.2 节的统一资源入口获取，也可按各实验的归档目录查阅。
