# HyperSynergyV4 OT-FM 代码交付审计

日期：2026-11-16（Asia/Shanghai）

**历史实现审计记录：代码实现完成，真实数据GPU验证暂缓。用户随后明确要求上传当前代码快照；下文“未提交/未推送”及REMOTE PUSH结论记录的是上传前阶段，实际远程状态以Git分支和最终上传回执为准。**
用户在 GPU 被其他用户进程占满后明确要求“先不进行训练，先把代码整体完成”。因此下面将已通过的工程检查与尚未执行的真实模型检查分开记录。

## 项目与 Git

- 基线：/data/jyk/HyperSynergyV2，使用当前完整工作区，包含原有未提交源码与未跟踪编码器，未将原始上游 HEAD 当成 V2。
- V4：/data/jyk/HyperSynergyV4。
- 基线历史 HEAD：f7e23e2b381b0120e88222124f0b06baac432630。
- V4 本地分支：feature/hyperV4-otfm；尚无本次提交，HEAD 仍是上述继承提交。
- V4 origin 已按用户新要求配置为 https://github.com/jolaaa999/HyperSynergyV4.git；只是本地配置，仓库未创建、未推送。
- GitHub 连接与本地 gh 均确认账号为 jolaaa999；目标查询返回404，未把该结果解释为已经建仓。
- V2 的分支、origin、工作区内容和 Git 状态均保持原样。
- V4 保留独立 Git 对象和历史，无共享可写工作区。作者身份仅写入 V4 的 .git/config，使用 GitHub noreply 地址。
- V4 的 .git/info/exclude 仅排除本地继承运行产物、checkpoint、锁和旧实验 CSV；没有修改原 .gitignore，没有删除这些历史文件。

## 基线审计

Stage1 为 representation_learning.py -> model_rep.EMBNet -> encoders.SampleFeatureExtractor，包含药物 TransformerConv/GraphNorm 和细胞 SE-CNN。

Stage2 为 train_few_shot.py。没有独立 resumable 入口、microbatch 或梯度累积代码。原最大预算为40000步（获授权早停允许提前结束）、batch50、Support50、Query40，外层 Adam lr=.001/weight_decay=1e-5，inner_l_rate 初始.1及其 Adam lr=.1，MultiStepLR 原里程碑和 gamma 不变。每200步按原 LOSS/best_r2 条件保存，验证规则未改。

Stage3 为 test_few_shot.py，原 batch50、Support5、Query40、200批、20次重复以及15步适应规则未改。暂缓的真实测试会在项目外限制运行任务数量，不能修改正式预算。

先验为高斯重参数化和源码原有 KL，无 LDM。原 KL 是 F.kl_div(h.softmax(-1).log(), noise.softmax(-1), reduction='mean')。原总损失为 Query_MSE + .001*KL + 1e-9*encoder_penalty，全部保留。

Inference_model 的卷积输出256维，分成128维均值/对数方差并高斯采样。meta_train_batch 完成15步Support适应后返回 h。该 h 被用于原权重生成网络和新增先验分支。

## OT-FM 实现

- Stage2 使用 h 作为 x1；按真实源码推导维度为 [B,128]。真实模型前向形状尚待 GPU 测试确认。
- x0 是同设备、同dtype、同形状的标准高斯噪声。
- 平方欧氏成本 C 为 [B,B]，Hungarian 求解一对一最小匹配。
- 只有匹配索引求解不求导；匹配目标从原始 h 用 index_select 取值，未 detach h。
- OT 重排只影响辅助先验，不改动 Query 潜变量、标签或样本顺序。
- t 为 [B,1]，线性插值 xt=(1-t)x0+t*x1，速度目标 x1-x0。
- 经用户明确批准，速度场为 concat(xt,t) -> 129/128/128/128，两个 SiLU，线性输出 [B,128]。
- 没有新增 LayerNorm、目标投影、维度压缩、Sinkhorn 或 ODE。
- 总损失仅增加 weight*FM，其中 FM 是对任务及维度的均值MSE。
- 默认关闭；正式 JSON 配置开启、weight=.02。零权重也不创建或运行FM分支。
- 在原优化器构建前注册新参数，原参数分组、优化器类型、学习率及调度器不变。
- Stage3 只增加新旧 checkpoint 识别和严格加载。代码中 eval 或 train1=False 不调用 FM。
- checkpoint 保存完整模型、先验元数据、双优化器、调度器及 RNG；拒绝未知/不完整元数据和缺失/多余模型参数，未使用 strict=False。
- 没有为原本不存在的 Stage2 resume 功能创建替代训练入口；状态恢复辅助接口已验证，完整续跑协议未擅自扩展。

## 文件变更

| 文件 | 类型 | 具体作用 |
|---|---|---|
| priors/__init__.py | 新增 | 独立辅助先验包 |
| priors/ot_flow_matching.py | 新增 | 成本、Hungarian、插值、时间条件速度场、FM均值MSE |
| prior_config.py | 新增 | 默认关闭、零权重行为、配置校验及JSON读取 |
| configs/hyperV4_otfm.json | 新增 | 仅 enabled=true 和 weight=.02 |
| otfm_checkpoint.py | 新增 | 显式新旧类型识别、严格加载、训练状态恢复 |
| docs/OTFM.md | 新增 | 算法、接口、配置、运行环境及限制说明 |
| docs/OTFM_IMPLEMENTATION_PLAN.md | 新增 | 实施和待验证计划 |
| docs/OTFM_AUDIT.md | 新增 | 本交付报告 |
| model_few.py | 修改 | HyperSynergy.__init__/configure_prior/run_batch，仅配置、注册及训练期加权辅助损失 |
| train_few_shot.py | 修改 | 可选配置解析、FM日志、checkpoint_payload保存；新增仅LOSS早停，其他原train规则保留 |
| test_few_shot.py | 修改 | test()的模型加载两行改为显式严格兼容层 |

除上述三个原有普通文件外，814个原有非Git普通文件摘要均一致。Stage1、model_rep.py、encoders/、data_syn.py、utils_syn.py、数据与历史权重均未改变。原网络及适应方法另做AST相等检查。

V4内33个历史绝对符号链接经用户批准重定向至对应V4目录；普通文件完整复制后先校验相等，再执行这项例外。V2内链接未改。

## 测试结果

| 检查 | 结果及范围 |
|---|---|
| 原始复制 | PASS：950条目、844文件，内容/权限/链接初始一致，Git存储独立 |
| OT成本与Hungarian | PASS：手算矩阵、一对一合法性、最优重排、成本不增，匹配确实用于FM |
| 路径和速度输入 | PASS：手算插值、速度方向、广播；hook核验MLP收到xt与t |
| 边界与设备 | PASS：CPU/GPU，B=1/2/50，有限标量MSE、目标梯度、速度参数梯度、无原地修改 |
| 配置和构造RNG | PASS：默认关闭和权重0不改变原参数或CPU构造RNG；开启初始化也保护原RNG |
| 严格兼容 | PASS：新旧checkpoint、损坏/缺失参数和元数据拒绝 |
| CPU文件恢复 | PASS：模型、双优化器、调度器、Python/NumPy/Torch RNG；Parameter身份和优化器引用保留，恢复后下一次合成单元更新一致 |
| 速度参数优化 | PASS：原参数收集方式包含全部FM参数，CPU合成输入单元检查确认权重实际更新 |
| 实际旧权重 | PASS：继承Stage1权重及3400k_0.0873旧Stage2 checkpoint strict=True加载、数值有限 |
| eval/train1=False | PASS，仅隔离分支单元检查；原CUDA依赖组件被替代，不能当作真实Stage3证据 |
| Python编译 | PASS：96个项目Python文件，未创建新缓存 |
| 独立代码复核 | 未发现Critical/Important问题；明确保留真实集成验证缺口 |
| 原网络/协议AST | PASS：推断/生成网络、Support更新/Query预测方法、扣除获授权早停后的训练循环和Stage3评估主体 |
| 本次差异空白检查 | PASS；相对上游有继承的representation_learning.py末尾空行警告，其SHA与V2相等，未修复 |
| 真实数据首次尝试 | FAILED：第一次数据上GPU即OOM，未进入模型前向；外部两个进程各占11618 MiB |
| 真实完整行为兼容 | DEFERRED：同初始权重/输入/RNG的前向、原损失、原梯度和单步更新尚未实测，不能声称逐位相等 |
| FM到Inference梯度及规模比较 | DEFERRED：目标h梯度单元检查通过，真实推断网络梯度范数尚未测量 |
| Stage2真实短运行 | DEFERRED：用户要求暂不训练 |
| Stage3真实任务评估 | DEFERRED：未产生预测或评估指标 |

上一阶段共11个单元测试方法通过（5个prior，2个配置/加载，4个CPU checkpoint/guard）；其中“真实数据”相关通过数为0，不能混淆。

CPU构造RNG与参数相等使用 torch.equal；CPU checkpoint及后续合成单元更新使用精确相等。真实数据兼容测试原定 rtol=1e-5 / atol=1e-6，但未到前向，故没有可报告的真实误差，也未放宽容差。

## 继承问题：仅记录

Stage2验证原来使用get_test_batch；GraphNorm原调用未传batch；冻结编码器后原train模式仍影响BatchNorm/Dropout；Support更新原有backward及grad.data副作用。均未修复或改变。

## 临时文件与证据

本阶段结束时删除本次新建测试脚本、测试日志、临时checkpoint、空sanity目录与review压缩包；不删除任何V2或继承的正式测试、历史运行结果与checkpoint。保留独立审计目录 /data/jyk/hyperV4_audit_20261116_cd9dly16 中的文件清单、SHA-256、批准链接映射、源码patch和结构化审计结果，用于后续补测。

## 实现阶段结论（上传前历史记录）

OT-FM IMPLEMENTATION: BLOCKED（代码已实现，完整真实模型验证未完成）
SOURCE ISOLATION: PASS（V2全部文件摘要及Git状态未变；V4变更全部可解释）
REMOTE PUSH: BLOCKED（依原门槛，全部测试通过前不提交、不建仓、不推送）

没有正式长时间训练；没有预测效果提升结论，FM Loss下降也不能作为效果提升证据。

## 2026-11-16 新增授权：Stage2 早停

按用户最新指示，早停仅以验证 LOSS 严格降低判断提升，与 checkpoint 判优独立。

- 仍在 step % 200 == 0 的原验证流程中检查。
- 单独记录历史最低 best_early_stopping_loss，初始为正无穷，第一次有限验证 LOSS 建立基线。
- 只有 LOSS < best_early_stopping_loss 才清零计数；相等、变差均累计一次。
- 连续5次未提升后，在完成当次验证日志记录后退出训练循环，不再进行下一步更新。
- R²提高不会重置早停计数。原checkpoint保存条件 LOSS降低或R²提高、文件选择、原best_loss/best_r2更新方式均保留。
- 不自动回滚权重；原40000步仍是上限，新增获授权早停可以提前结束。
- V2、原网络、损失、学习率、采样、Stage3均未修改。

本次7个CPU控制流程单元测试通过：第5次失败边界、LOSS改善重置、R²改善不能重置、未满5次不停止、首个有限LOSS独立建立基线、checkpoint更新不能覆盖历史早停最优值、持续改善达到预算。测试运行原train函数并替代数据/模型/I/O，仅验证控制流程，不是真实训练，也不是预测效果证据。

本次仅修改train_few_shot.py及相关说明文档。相比上一交付，训练协议唯一新增变化是上述经用户明确批准的早停。全部真实数据集成测试仍待执行；本次没有训练、提交、建仓或推送。临时测试脚本和日志按精确路径删除，保留源码差异及结构化审计证据。

## 最新上传授权

用户在明确获知真实数据测试尚未完成、代码尚未上传后，最新指示“上传远程仓库”。按该新指示提交当前源码快照至 jolaaa999/HyperSynergyV4 的 feature/hyperV4-otfm 分支，保留全部待验证说明；未运行正式训练，未将待验证项标记为通过。README仅增加版本说明，原README正文保留。

## 2026-11-16 FM系数更新

按用户新指示，当前PriorConfig默认weight与configs/hyperV4_otfm.json的weight均由0.02改为0.005。上文0.02为历史实施/验证时的初始设置。OT-FM默认仍关闭，正式V4配置仍开启；原KL、encoder penalty系数、学习率、早停和其他模型逻辑均不变。不改写已有checkpoint中的历史元数据；本次未启动训练。
