# PhiStone 数据准备与预处理流程

## 1. 数据准备概述与技术路线

本文档介绍 PhiStone 竞赛项目的数据来源、文献提取、数据清洗、图数据构建、轻量词嵌入模型训练及生成片段词表准备流程，为后续活性预测与分子生成模型提供数据准备依据。

项目数据准备包括两条主要路线：**抗结核小分子活性数据集构建**与**配体—蛋白复合物数据集构建**。两条路线分别生成二维和三维 PyTorch Geometric（PyG）图数据，并在预处理过程中整理共享粗语料，用于训练片段编码器与原子编码器。

### 1.1 数据体系与总体流程

| 数据路线 | 数据来源 | 主要处理 | 主要产出与用途 |
| --- | --- | --- | --- |
| 抗结核小分子活性数据 | 文献 PDF 中的分子结构、活性数值、测试对象及来源信息 | 文献提取与修复、菌株筛选、活性标准化、分子去重、二维图构建 | 分子级活性表格与二维 PyG 图数据，用于后续活性模型训练。 |
| 配体—蛋白复合物数据 | HiQBind 与 PDBbind 原始复合物 | 结构检查、清洗去重、蛋白口袋整理、三维图构建 | 含配体与蛋白口袋空间信息的三维 PyG 图数据，用于 PhiSSE3TD 训练。 |
| 共享语料与词嵌入 | 二维、三维预处理产生的片段及片段内原子粗语料 | 轻量编码器训练、词嵌入表导出、生成片段词表构建 | 128 维片段嵌入、64 维原子嵌入及编码器权重，供后续训练与推理使用。 |

```text
抗结核小分子活性数据路线
文献 PDF
  → 分子—活性记录提取与修复
  → 目标菌株筛选与未测试记录清理
  → 活性标准化、分子去重与活性汇总
  → 二维分子图构建

配体—蛋白复合物数据路线
HiQBind + PDBbind 原始复合物
  → 结构检查、清洗与重复复合物合并
  → 配体周围蛋白口袋整理
  → 三维复合物图构建

共同后续流程
二维 / 三维 PyG 图数据 → 质量检查与清理 → 后续模型使用的图样本
二维 / 三维预处理产生的共享粗语料
  ├── 片段编码器训练 → 128 维片段词嵌入表
  └── 原子编码器训练 → 64 维原子词嵌入表
标准词表与编码器权重
  → 中性片段扩充（可选）
  → 可解码生成片段词表筛选
  → 后续分子生成流程
```

### 1.2 路径与复现约定

本文中的项目路径均相对于 `PhiStone_2.0_UP/` 根目录。命令示例应在已配置项目依赖的环境中，于项目根目录下按需执行。

文献 PDF、原始表格与处理记录、已处理的二维图数据、已处理的三维图数据及共享数据资源的获取方式，分别列于第 2、3、5 节。资源文件名用于标识网盘共享文件；项目内的输入输出位置以各处理阶段的路径表为准。

PhiSGATv2、PhiSSE3TD 的训练、微调及交互式推理入口见 [README.md](README.md)。

## 2. 抗结核小分子活性数据集构建

该路线从文献中提取分子结构及活性信息，经目标菌株筛选、单位标准化和分子级归并，形成活性训练标签，并进一步构建二维分子图数据。除综合利用 MIC、IC50 和抑制率信息的主数据路线外，项目还保留仅使用 MIC 信息的消融实验测试组。

### 2.1 文献提取与结果修复

文献提取阶段整理分子结构、活性数值、测试对象及文献来源，结果修复阶段结合原文对提取记录进行补充和修正，形成供后续清洗使用的原始分子—活性表格。

项目使用的文献 PDF 可通过以下资源获取：

| 资源名称 | 资源用途 | 下载链接 | 提取码 |
| --- | --- | --- | --- |
| `PDF` | 文献提取与结果修复所使用的文献。 | [百度网盘](https://pan.baidu.com/s/14XNbjvPLEqMcPppkhdY4-Q?pwd=cgwk) | `cgwk` |

| 脚本位置 | 主要功能 |
| --- | --- |
| `Molecule extraction/Extractor_Gemini.py` | 从文献中提取分子及其活性关系，整理为表格记录。 |
| `Molecule extraction/Repair_Extractor.py` | 结合原文对已有提取结果进行补充与修复。 |
| `Molecule extraction/Extractor.py` | 整合文献提取与结果修复功能。 |

本次数据构建使用 `Extractor_Gemini.py` 与 `Repair_Extractor.py` 完成提取和修复，`Extractor.py` 作为整合脚本保留。上述三个工具不构成必须依次执行的处理链。原始 PDF、提取结果与修复结果的保存位置由相应脚本配置。

### 2.2 数据清洗与初步筛选

抗结核小分子活性数据处理得到的原始表格文件和处理记录可通过以下资源获取：

| 资源名称 | 资源用途 | 下载链接 | 提取码 |
| --- | --- | --- | --- |
| `Molecule extraction.tar` | 抗结核小分子活性数据的原始表格文件与处理记录。 | [百度网盘](https://pan.baidu.com/s/1HLfAPua_4KgjO3mV9MdtsQ?pwd=z3dm) | `z3dm` |

清洗脚本位于 `Molecule extraction/Data_Process/origindata/`。按照输入依赖，先提取目标菌株名单，再筛选和重新归类活性记录，最后清理仅包含未测试占位值的记录。

| 顺序 | 脚本 | 主要处理 | 默认输出 |
| --- | --- | --- | --- |
| 1 | `extract_hazard_a_target_strains.py` | 从 Hazard Level 为 A 的记录中提取不重复的 Target Strain，形成筛选名单。 | `hazard_A_target_strains.txt` |
| 2 | `filter_human_mtb_activity_rows.py` | 按名单匹配目标菌株，保留相关活性记录并重新归类；分别处理空白或无法分类的活性。 | `human_mtb_activity_reclassified.xlsx` |
| 3 | `clean_untested_activity.py` | 剔除所提供活性测试值仅为 `-`、`nt`、`nd` 等未测试占位内容的记录。 | `human_mtb_activity_reclassified_cleaned.xlsx` |

上述输出默认保存在 `origindata/` 目录。清洗结果随后需汇总整理为 `Molecule extraction/Data_Process/ALLIN/activity_review.xlsx`，作为活性标准化阶段的默认输入。该汇总整理是独立的数据衔接步骤，不由上述三个脚本自动串联完成。

### 2.3 活性标准化与分子去重

主数据路线的标准化脚本及表格位于 `Molecule extraction/Data_Process/ALLIN/`。

| 脚本 | 默认输入 | 主要处理 | 默认输出 |
| --- | --- | --- | --- |
| `standardize_mic_level.py` | `activity_review.xlsx` | 按脚本规定的优先级解释 IC50、抑制率与 MIC 信息，统一可解析数值的单位，并赋予项目自定义 Activity Level。 | `activity_mic_level.xlsx` |
| `standardize_mic_level_mean.py` | `activity_mic_level.xlsx` | 规范化 SMILES，以分子为单位归并重复记录，按既定规则筛选差异较大的活性记录，再汇总均值。 | `activity_mic_level_mean.xlsx` |

标准化过程中，IC50 与 MIC 的浓度单位统一为 μM，测试浓度使用 μg/mL；IC50 不直接换算为 MIC。分子去重采用规范异构 SMILES，使等价写法合并，同时保留不同立体化学结构的区分。

最终分子级活性表格保存于：

```text
Molecule extraction/Data_Process/ALLIN/activity_mic_level_mean.xlsx
```

主要字段包括 `Compound ID`、`Canonical SMILES` 和 `Mean Activity Level`。其中，`Mean Activity Level` 是按项目规则定义并汇总的训练标签，不等同于原始实验测得的 MIC 浓度。

### 2.4 仅使用 MIC 数据的消融实验测试组

MIC 数据支线用于与综合利用 MIC、IC50 和抑制率信息的主数据路线进行对照。相关脚本位于 `Molecule extraction/Data_Process/MICONLY/`，按照先标准化、再归并分子的顺序处理。

| 顺序 | 脚本 | 主要处理 | 默认输出 |
| --- | --- | --- | --- |
| 1 | `standardize_mic_pmic.py` | 筛选具有 MIC 记录的数据，统一可转换 MIC 的单位，并计算脚本定义的标准化活性分数 `Standardized pMIC`。 | `human_mtb_activity_reclassified_cleaned_pMIC.xlsx` |
| 2 | `aggregate_unique_smiles_activity.py` | 规范化并去重 SMILES，按测量一致性规则筛选记录，汇总分子级活性均值。 | `human_mtb_unique_smiles_activity_mean.xlsx` |

第一步默认输入为 `human_mtb_activity_reclassified_cleaned.xlsx`，需将清洗结果整理至该支线的输入位置。最终表格保存于：

```text
Molecule extraction/Data_Process/MICONLY/human_mtb_unique_smiles_activity_mean.xlsx
```

该测试组仅以 MIC 作为活性评分依据，不使用 IC50 或抑制率补充评分，并与 `ALLIN/activity_mic_level_mean.xlsx` 主数据集分别保留。

### 2.5 二维分子图构建与已处理数据获取

`PhiSSeparator/PhiSSeparator_SE3TD_2D.py` 读取最终分子及活性标签，构建后续活性模型使用的二维 PyG 图数据，并整理共享粗语料。

最终表格的产出位置与二维预处理的默认输入位置不同。复现时需将表格整理至默认输入位置，或调整脚本的输入配置。

| 数据或处理入口 | 项目相对路径 |
| --- | --- |
| 最终表格产出位置 | `Molecule extraction/Data_Process/ALLIN/activity_mic_level_mean.xlsx` |
| 二维预处理默认输入 | `Datas/Pre_datas/Antitubercular Molecules/activity_mic_level_mean.xlsx` |
| 二维预处理脚本 | `PhiSSeparator/PhiSSeparator_SE3TD_2D.py` |
| 二维 PyG 图输出 | `Datas/Processed_datas/SE3TD_2D_128_Pygdata/pyg_pending/` |
| 共享数据目录 | `Datas/Processed_datas/SE3TD_128_Shared_data/` |

本次处理完成的抗结核小分子二维图数据可通过以下资源获取：

| 资源名称 | 资源用途 | 下载链接 | 提取码 |
| --- | --- | --- | --- |
| `SE3TD_2D_128_Pygdata.tar.gz` | 已处理的抗结核小分子二维 PyG 图数据。 | [百度网盘](https://pan.baidu.com/s/1TOOBEJgyDfKUvurT7jIkaA?pwd=gyj6) | `gyj6` |

图数据的质量检查与清理见第 4 节，共享语料的轻量编码器训练见第 5 节。

## 3. 配体—蛋白复合物数据集构建

复合物数据来自 **HiQBind** 与 **PDBbind**，用于构建包含配体及蛋白口袋空间信息的三维训练数据。该路线先检查和清洗原始结构，合并重复复合物并整理蛋白口袋，再构建三维 PyG 图数据与共享粗语料。

### 3.1 原始结构清洗、去重与口袋构建

处理脚本为 `Datas/Pre_datas/prepare_hiqbind_pdbbind.py`。该脚本检查配体和受体结构，过滤不合格记录，合并两个数据来源中的重复复合物，并整理配体周围的蛋白口袋。

去重采用 **PDB ID 与经过电荷、质子化规范化的配体异构 SMILES** 作为联合依据。对于重复组，当 HiQBind 的配体与受体均通过检查时，优先保留 HiQBind 记录。

默认口袋截取范围为配体各原子周围 **10 Å**：受体残基只要有任一原子进入该范围，即保留整个残基；默认去除水。该规则用于训练复合物数据的蛋白口袋构建。

清洗结果默认保存于：

```text
Datas/Processed_datas/HiQBind_PDBbind_clean/
├── accepted_manifest.csv
└── complexes/
```

其中，`accepted_manifest.csv` 记录通过检查的复合物清单，`complexes/` 保存整理后的复合物文件。

### 3.2 三维图构建与已处理数据获取

`PhiSSeparator/PhiSSeparator_SE3TD_3D.py` 读取清洗结果及通过检查的复合物清单，对配体与蛋白口袋进行片段化和特征整理，构建后续 PhiSSE3TD 训练使用的三维 PyG 图数据。

| 数据或处理入口 | 项目相对路径 |
| --- | --- |
| 清洗结果根目录 | `Datas/Processed_datas/HiQBind_PDBbind_clean/` |
| 通过检查的复合物清单 | `Datas/Processed_datas/HiQBind_PDBbind_clean/accepted_manifest.csv` |
| 整理后的复合物文件 | `Datas/Processed_datas/HiQBind_PDBbind_clean/complexes/` |
| 三维预处理脚本 | `PhiSSeparator/PhiSSeparator_SE3TD_3D.py` |
| 三维 PyG 图输出 | `Datas/Processed_datas/SE3TD_HiQBind_PDBbind_128_Pygdata/pyg_pending/` |
| 共享数据目录 | `Datas/Processed_datas/SE3TD_128_Shared_data/` |

本次处理完成的配体—蛋白复合物三维图数据可通过以下资源获取：

| 资源名称 | 资源用途 | 下载链接 | 提取码 |
| --- | --- | --- | --- |
| `SE3TD_HiQBind_PDBbind_128_Pygdata.tar.gz` | 已处理的 HiQBind、PDBbind 配体—蛋白复合物三维 PyG 图数据。 | [百度网盘](https://pan.baidu.com/s/1BwZMMqOZBc5RLY5t7vz2jw?pwd=ftw1) | `ftw1` |

完成三维预处理后，按照第 4 节检查图数据，并按照第 5 节准备共享词嵌入。

## 4. 图数据质量检查与清理

### 4.1 结构抽样检查与规模统计

完成二维和三维预处理后，分别使用对应 Notebook 检查生成的 `.pt` 图数据文件。

| 检查入口 | 对应数据目录 | 主要检查内容 |
| --- | --- | --- |
| `PhiSSeparator/inspect_pyg_samples_2D.ipynb` | `Datas/Processed_datas/SE3TD_2D_128_Pygdata/pyg_pending/` | 随机抽样展示完整分子和片段结构网格，查看节点、共价边、活性标签及对象属性。 |
| `PhiSSeparator/inspect_pyg_samples_3D.ipynb` | `Datas/Processed_datas/SE3TD_HiQBind_PDBbind_128_Pygdata/pyg_pending/` | 随机抽样查看节点、空间边、配体片段连接及对象属性，展示反向归一化的节点特征。 |

两个 Notebook 均提供全量节点数、边数统计及分布图，用于识别规模异常大或异常小的样本。清理依据为**节点数和去重后的无向边数**，保留范围包含边界值，不以磁盘文件字节大小作为筛选依据。

### 4.2 异常样本隔离

`PERFORM_CLEANING` 控制是否执行清理。2D 检查 Notebook 中该开关为 `True`，3D 为 `False`；复现时应先核对实际配置。建议按照以下顺序操作：

1. 将 `PERFORM_CLEANING` 设为 `False`，查看抽样结构与规模统计。
2. 根据节点数和无向边数分布确定样本保留范围。
3. 按需将 `PERFORM_CLEANING` 设为 `True`，执行异常样本隔离。

超出保留范围的 `.pt` 文件将移至与各自 `pyg_pending/` 同级的 `pyg_pending_quarantine/` 目录，保留样本仍位于 `pyg_pending/`，供后续训练读取。清理采用文件隔离方式，不直接删除样本，也不会自动重新生成共享粗语料或词嵌入表。

## 5. 轻量词嵌入模型训练

二维和三维预处理脚本在图数据构建过程中整理粗片段语料与片段内原子语料。项目分别训练片段编码器和原子编码器，将其转换为数值表示：**每个片段对应 128 维嵌入，每个片段内的原子对应 64 维嵌入**。

### 5.1 共享语料与数据资源

两类粗语料及导出的标准词嵌入表均保存在：

```text
Datas/Processed_datas/SE3TD_128_Shared_data/
```

本阶段使用的共享数据资源可通过以下文件获取：

| 资源名称 | 资源用途 | 下载链接 | 提取码 |
| --- | --- | --- | --- |
| `SE3TD_128_Shared_data.tar.gz` | 轻量词嵌入模型训练相关的共享数据资源。 | [百度网盘](https://pan.baidu.com/s/1SVsO-ZqwcmFq6s9bArNZIw?pwd=ckkm) | `ckkm` |

下述文件表列出项目使用的粗语料、元数据、标准词表及编码器权重位置。编码器权重保存于 `PhiSSeparator/training_artifacts/`。

### 5.2 片段编码器训练：128 维

训练入口为 `PhiSSeparator/PhiSSeparator_fragment_encoder_train.py`。脚本读取共享目录中的粗片段语料，训练片段编码器，并使用最佳权重对完整粗语料进行编码，导出标准片段词嵌入表。

| 文件类型 | 默认文件位置 |
| --- | --- |
| 粗片段语料 | 共享目录下的 `fragment_raw_corpus.npz` |
| 粗语料元数据 | 共享目录下的 `fragment_raw_corpus_metadata.json` |
| 最佳编码器权重 | `PhiSSeparator/training_artifacts/fragment_encoder_128.pth` |
| 128 维片段词嵌入表 | 共享目录下的 `fragment_embeddings_128d.npz` |
| 词表元数据 | 共享目录下的 `fragment_embeddings_128d_metadata.json` |

脚本同时生成可读 JSON 词表镜像及训练记录。嵌入不会回填至已有 PyG 文件，而由后续流程加载词表及编码器权重使用。

### 5.3 原子编码器训练：64 维

训练入口为 `PhiSSeparator/PhiSSeparator_atom_encoder_train.py`。脚本利用片段内部的原子及键环境训练原子编码器，并使用最佳权重导出标准原子词嵌入表。原子表示包含片段环境信息，并非仅按元素种类建立查找表。

| 文件类型 | 默认文件位置 |
| --- | --- |
| 片段内原子粗语料 | 共享目录下的 `fragment_atom_raw_corpus.npz` |
| 粗语料元数据 | 共享目录下的 `fragment_atom_raw_corpus_metadata.json` |
| 最佳编码器权重 | `PhiSSeparator/training_artifacts/atom_encoder_64.pth` |
| 64 维原子词嵌入表 | 共享目录下的 `fragment_atom_embeddings_64d.npz` |
| 词表元数据 | 共享目录下的 `fragment_atom_embeddings_64d_metadata.json` |

脚本同时生成可读 JSON 词表镜像及训练记录。复现时，先完成所需数据的预处理、粗语料整理和图样本检查，再分别训练两个轻量编码器，并保留导出的词表、元数据及最佳权重，供后续模型训练与推理使用。

## 6. 生成片段词表构建

在标准片段词表、原子词表及两个编码器权重准备完成后，可进一步扩充带电片段的中性对应物，并筛选可用于生成流程的片段候选。需要使用新增中性条目时，应先完成中和扩充，再进行可解码词表筛选。

### 6.1 带电片段的中性对应物扩充

处理脚本为 `PhiSSeparator/PhiSSeparator_neutral_fragment_vocab_extender.py`。脚本扫描两个标准词表中的 SMILES，对**总形式电荷非零**的片段尝试中和；总电荷已为零的结构不进入中和步骤。

中和后的结构需同时满足以下条件，才会交由原子和片段编码器的 resolver 处理：

- 总形式电荷为零。
- 重原子数量保持不变。
- 能够重新解析为分子结构。

原始带电条目保留，新增中性条目使用已有编码器权重计算嵌入，无需重新训练模型。resolver 负责追加共享粗语料、标准词嵌入表、对应元数据及 JSON 镜像。中和过程会去除立体化学信息，因此新增中性 SMILES 不保留原结构的立体标记。

运行前需准备第 5.2、5.3 节列出的两套粗语料、元数据、最佳权重及标准词表。脚本支持 `--device` 指定编码设备；`--dry-run` 仅扫描并输出中和映射，不实例化 resolver，也不写入词表。

```bash
# 查看中和映射，不写入词表
python PhiSSeparator/PhiSSeparator_neutral_fragment_vocab_extender.py --dry-run

# 追加中性条目，默认自动选择设备
python PhiSSeparator/PhiSSeparator_neutral_fragment_vocab_extender.py
```

脚本输出扫描统计、中和映射、无法中和的条目及 resolver 处理结果。

### 6.2 可解码生成片段筛选

处理脚本为 `PhiSLinker/build_decodable_fragment_vocabulary.py`。脚本从共享目录中的 `fragment_embeddings_128d.npz` 读取标准片段词表，筛选适合作为生成候选且能够构建三维构象的片段。

筛选规则如下：

- 总形式电荷为零，元素仅限 H、C、N、O、F、P、S、Cl、Br、I。
- RDKit 感知的环数不超过 4，且不含三元环、四元环或桥环结构。
- 调用 `PhiSLinker_atom_pair_smiles_decoder.py` 中的 `build_fragment_conformer_template`，能够生成坐标值均为有限数值的三维构象。

三维坐标仅用于判断片段能否解码，不写入输出词表，也不进行参考坐标系对齐。输出保留通过筛选的原始 SMILES 及其原有 128 维嵌入，不重新训练或计算片段嵌入。

默认输出目录为：

```text
Datas/Processed_datas/SE3TD_128_Shared_data/fragment_custom_embedding_128d/
```

| 输出文件 | 内容与用途 |
| --- | --- |
| `fragment_embeddings_128d.npz` | 全部通过筛选的可解码片段。 |
| `fragment_survival_linker_embeddings_128d.npz` | 通过筛选且重原子数不超过 5 的片段，供 `survival_linker` 使用。 |
| `fragment_activity_modify_embeddings_128d.npz` | 通过筛选且重原子数不超过 6 的片段，供 `activity_modify` 使用。 |
| `fragment_survival_modify_embeddings_128d.npz` | 通过筛选且重原子数不超过 7 的片段，供 `survival_modify` 使用。 |
| `fragment_decode_report.json` | 保留与排除数量、原始行索引及排除原因等报告。 |

运行环境需提供 NumPy 和 RDKit。底层解码优先使用 RDKit；构象生成失败时尝试调用 `obabel`。如需该回退能力，服务器环境应提供可用的 Open Babel。回退仍失败的片段将记录为解码失败并排除。

### 6.3 命令示例与后续衔接

```bash
# 首次生成，默认不覆盖已有输出
python PhiSLinker/build_decodable_fragment_vocabulary.py

# 标准词表扩充后，重新筛选并替换已有输出
python PhiSLinker/build_decodable_fragment_vocabulary.py --overwrite
```

可通过 `--input-vocab` 与 `--output-dir` 指定输入词表和输出目录。默认目录中已有任一目标输出时，未指定 `--overwrite` 的运行会停止并提示。筛选结果写入独立目录，不覆盖输入的标准片段词表。

生成片段词表的衔接顺序为：**二维或三维预处理 → 共享粗语料整理 → 两个轻量编码器训练 → 标准词表中性扩充（可选） → 可解码生成片段词表筛选 → 后续分子生成流程使用**。

经检查与清理的图样本、标准词嵌入表及编码器权重用于对应模型的训练与推理，生成片段词表用于后续生成模块的片段候选选择。后续 PhiSGATv2 训练、PhiSSE3TD 基础训练与微调，以及交互式分子生成流程见 [README.md](README.md)。
