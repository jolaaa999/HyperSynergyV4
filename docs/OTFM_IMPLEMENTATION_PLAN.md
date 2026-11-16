# HyperSynergyV4 OT-FM implementation plan

Spec: user implementation request, 2026-11-16; baseline /data/jyk/HyperSynergyV2 current working tree.
Goal: add a training-only OT-FM auxiliary prior while preserving every V2 loss and prediction protocol.
Architecture: Hungarian assignment on squared Euclidean Gaussian-to-adapted-latent costs; linear FM path; concat(x_t,t) -> 129/128/128/128 MLP with two SiLU activations.
Constraints: lambda=0.02 initially; default disabled; no V2 writes, encoder/inner-loop/sampling/optimizer/LR/selection changes, dependencies, ODE, normalization or target projection.
Approved copy exception: 33 historical symlinks retargeted within V4. All regular files verified equal before changes. No shared Git storage.

## Task 1: isolated copy (complete)
- [x] Hash all files including ignored files and Git metadata before/after copy; compare permissions and paths.
- [x] Copy with cp -a and preserve independent Git history; retarget only approved links.

## Task 2: prior implementation
Files: priors/__init__.py, priors/ot_flow_matching.py, prior_config.py, configs/hyperV4_otfm.json.
Interfaces: squared_cost(x0,x1), optimal_transport_pair(x0,x1), flow_path(x0,x1,t), OTFlowMatchingPrior.forward(h).
- [x] Run failing unit tests first; then implement exact Hungarian + FM with graph-preserving target indexing.
- [x] Verify CPU/GPU, B=1 and B=50, formulas, assignment, finite loss, gradients and no mutation.

## Task 3: integration and strict checkpoints
Files: model_few.py, train_few_shot.py, test_few_shot.py, otfm_checkpoint.py.
Interfaces: HyperSynergy(..., prior_config=None), configure_prior(config), last_fm_loss; checkpoint_payload and load_checkpoint.
- [x] Test missing configuration/checkpoint support before modifying model.
- [x] Add FM after query forward only in training; preserve six-value return and every original loss.
- [x] Add optional --prior-config; register velocity parameters before unchanged optimizer construction.
- [x] Save prior metadata and full strict model state; strict legacy/V4 loading, never strict=False.
- [x] Verify optimizer/scheduler state round trip using checkpoint helpers. Original entry had no Stage2 resume CLI; do not add a new training entry.

## Task 4: real-data verification and final audit
- [ ] Compare V2/V4 disabled and lambda=0 predictions, losses, RNG, gradients and one-step updates under identical state.
- [ ] Test real Stage2 steps, FM/query inference gradient norms, frozen Stage1, B=50 memory boundary, strict Stage1/legacy/V4 checkpoint loading and Stage3 5-support/40-query adapted evaluation.
- [ ] Preserve original 15-step adaptation, validation partition selection, metrics and budgets, including inherited issues.
- [ ] Remove only new temporary test files/checkpoints/logs; retain final audit evidence.
- [ ] Rehash V2 and V4, enumerate all changes, AST/config/interface/diff checks and independent review.
- [ ] Only after every required check passes, commit and create/push jolaaa999/HyperSynergyV4; verify remote hash.

Review focus: hidden RNG consumption with lambda=0; target gradient detached by matching; FM executed during Stage3; old checkpoints silently ignored; inherited file mutation or absolute links reaching V2.

Status: implementation complete; real-data GPU verification deferred by user after external GPU occupancy blocked the first tensor transfer. The user subsequently explicitly requested uploading this code snapshot with deferred checks disclosed. This upload does not mark those checks as passing.

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

Upload amendment: latest user instruction authorizes committing and publishing the current implementation snapshot before deferred real-data tests; preserve their pending status and all source-isolation requirements.
