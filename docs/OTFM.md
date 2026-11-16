# HyperSynergyV4 OT-FM

本版本以 /data/jyk/HyperSynergyV2 当前完整工作区为唯一基线，保留其独立 Git 历史及未提交的 DiffPro 编码器替换。没有修改 V2。

## 算法和接口

Stage2 的 model_few.HyperSynergy.run_batch 在原 Query 预测及原损失计算后，使用 meta_train_batch 返回的 Support 适应后任务潜变量 h，形状 [B,128]，计算训练期辅助先验。原 Inference_model、15 步 Support 更新、权重生成网络和 Query 对应顺序均不变。

1. x0 ~ N(0,I)，形状与 h 相同。
2. 计算 C[i,j] = sum((x0[i]-h[j])**2)，用 scipy.optimize.linear_sum_assignment 做一对一 Hungarian 匹配。只重排先验分支；索引求解停止梯度，但匹配目标从原始 h 取值，保留梯度。
3. 每任务采样 t ~ U(0,1)，形状 [B,1]；xt=(1-t)*x0+t*x1，u=x1-x0。
4. 速度场输入 concat(xt,t)，不是 concat(h,t)。MLP: 129->128->128->128，两个隐藏层使用 SiLU，最后一层线性；输出 [B,128]。
5. FM 为任务和维度上的均值 MSE。
6. total = Query_MSE + 0.001*原KL + 1e-9*原encoder_penalty + weight*FM。

原 KL 保持源码定义：F.kl_div(h.softmax(-1).log(), gaussian_noise.softmax(-1), reduction='mean')，不是标准闭式 Gaussian KL。无 LDM、LayerNorm、目标投影或 ODE。速度场要求实际维度等于128，不兼容则报错。

## 配置

默认 HyperSynergy(num_support,num_query) 完全关闭新先验。PriorConfig(enabled=True,weight=0) 同样不创建速度场、不采样 FM 噪声或时间。可选分支初始化使用 fork_rng 保护原模型随机状态。

正式 V4 配置 configs/hyperV4_otfm.json：
- enabled: true
- weight: 0.005

0.005 仅是初始实验值，尚未证明最优；没有更改任何原损失系数或实验超参数。配置使用 JSON，避免添加 YAML 依赖。

Stage2 原入口新增 --prior-config 参数，例如：
    python -B train_few_shot.py --prior-config configs/hyperV4_otfm.json

这是正式入口示例，本次未启动正式训练。目标环境沿用 Python3.6 / torch1.7.0+cu110 / scipy1.5.4 / PyG1.7.2。RTX4090 运行时兼容设置沿用原 run_v2_training.sh 中三个禁用 JIT fusion 的开关；该继承启动脚本未改动，其原有拒绝覆盖权重检查仍有效。

## 优化器和 checkpoint

速度场是模型注册子模块，构造发生在原优化器参数收集之前，因此进入原 Adam 参数组。两个原优化器、学习率及调度器设置不变。

otfm_checkpoint.checkpoint_payload 保存完整严格模型状态、显式 prior_format/prior_config，以及传入的双优化器、调度器和 RNG 状态。load_checkpoint 支持：
- 无先验元数据的原 V2 checkpoint：识别为原路径并 strict=True 加载。
- 有完整 V4 元数据的 checkpoint：创建相同先验并 strict=True 加载。
- 不完整/不认识的元数据、无元数据的 FM 权重或缺失/多余模型参数：明确报错。
- 恢复优化器时须先按 checkpoint 配置构造模型与优化器；相同配置不会重建 Parameter 对象。
- restore_rng=True 可恢复已保存的随机状态。

没有创建新的可恢复训练入口；原 Stage2 本来也没有该入口。辅助函数支持状态恢复，完整正式训练续跑仍须遵守原日志和验证状态协议，不能只根据 step 擅自重启。

Stage3 只增加显式 checkpoint 识别和严格加载，继续执行原有预测、任务采样、15 步 Support 适应及指标汇总。eval 或 train1=False 时不计算 FM，不匹配 OT，不做 ODE。

## 验证状态

已完成基础 OT/FM CPU/GPU 单元测试、CPU 配置/RNG/checkpoint 测试、实际继承 Stage1/Stage2 权重严格加载、独立静态代码复核及源码隔离检查。

真实数据测试在第一次数据传入 GPU 时因其他用户进程占满显存而失败，尚未执行真实模型前向。用户随后明确要求暂不训练。因此真实推断网络梯度规模、V2/V4 完整前向/梯度/单步更新一致性、正式 batch=50 的短 Stage2、实际 Stage3 评估仍待执行；不能声称这些已通过。

CPU eval/train1=False 的禁止 FM 调用检查仅为隔离分支的单元测试，其中 CUDA 依赖的原组件被替代，不等同实际 Stage3 验证。

用户在明确知悉真实数据验证暂缓的状态后，最新要求上传本次代码快照至 jolaaa999/HyperSynergyV4。该指示仅允许带待验证说明上传，不表示缺失测试已经通过。仅工程验证通过也不意味着预测效果提升。

## 继承问题（仅记录，未修复）

- 原 Stage2 验证调用 get_test_batch，而不是 get_val_batch。
- 原药物编码器 GraphNorm 调用未传 batch。
- 冻结编码器参数后，原 model.train() 仍会切换 BatchNorm/Dropout 模式。
- 原 Support 内循环有 backward 和 grad.data 更新及原有梯度副作用。

所有这些均按用户要求原样保留。历史运行目录仅用于封存；其中33个指向V2的绝对符号链接已在V4内重定向至对应V4路径，详见审计记录。

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

## 本次上传说明

本仓库发布当前完整V4源码和继承Git历史，真实数据完整前向/梯度/参数更新兼容、FM推断网络梯度规模及实际Stage2/Stage3短运行仍待执行。只上传源码、配置、文档和原有Git跟踪资源；本地历史训练产物未入库。
