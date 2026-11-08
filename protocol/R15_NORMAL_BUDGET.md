# R15：正常训练预算下的外部反证实验

日期：2026-09-17。此协议在本轮任何正式训练及测试评估前拟定；最终冻结以 `configs/r15_normal_budget/manifest.json` 的 UTC 和源码/本协议 SHA256 为准。旧 R9/R12/R13 raw、协议及投稿包不修改。新实验既可支持也可推翻已有观察，不把更多训练自动解释为更强贡献。

## 问题和与已有研究的关系

现有 CIFAR-10 的 actual/probe 排序分歧及候选类敏感性是否仅属于小模型、短训练情形？唯一新增实验采用 CIFAR-100、残差网络和有文献实现来源的完整训练轮数，检验同一联合模式是否延续。它是外部条件反证检验，不是新的 plasticity 定义、选择原则、机制或 Inamdar/Cho 官方复现。正常训练轮数不等于已证明优化收敛，也不保证排除所有欠训练解释。

## 固定矩阵

- 五个配对新运行种子：3101、3102、3103、3104、3105；两条历史：SGD、ER。
- 官方 CIFAR-100：50,000 train，10,000 test。已有项目已查看该 test；新种子及新训练并不恢复未见数据集身份。测试不选超参数、任务、种子或终点。
- `PCG64(seed+40000)` 类排列；十任务，每任务十类。网络初始 seed+10000，沿用固定 PyCIL ResNet-32、八组 GN 与100输出 head（470,004参数）。沿用 `experiments/r9_external/model.py` 与其有许可证的固定上游源。
- 训练仅 softmax 已见的 `10*(task+1)` 输出。每任务先将新激活的十个 head 行恢复为配对初始化，再保存 pre-state；其余参数继承 actual 历史。只有 actual 推进历史。
- ER 每个旧类保存20个随机选定的训练索引，沿用既有固定 class-specific 优先排列；无在线 reservoir 写入。每步 replay 数等于 current 数，旧 memory 每轮独立排列并按需循环；current/replay 拼接后平均 CE。SGD 不使用 memory。
- 方法与分支匹配 optimizer 更新数和 current 样本曝光；ER 额外处理 replay，故总样本曝光及计算量不匹配。
- Current batch64，末批8，每轮79次更新；首次任务200轮，后续每任务70轮。SGD momentum0.9，dampening0，无 Nesterov；初始 LR0.1。首任务完成60/120/170轮后各乘0.1，weight decay0.0005；后续及辅助分支完成30/50轮后各乘0.1，weight decay0.0002。所有参数包含GN affine/head bias使用相同coupled decay。每任务/分支重建 optimizer，momentum归零。
- 数值轮数/优化器来源为已核验 PyCIL replay 设置（见 `protocol/R9_EXTERNAL_MEMORY_DOSE_DRAFT.md` §4/6）。本实验以 current epoch 计数、GN、固定 current/replay 比例和统一新 head 初始化，区别于上游 union loader/BN 实现。
- 训练增广：四像素零 padding 后随机32×32 crop、0.5 horizontal flip、连续均匀 brightness factor `1±63/255`，截断至[0,1]，再按 CIFAR-100 mean/std normalize。采用批量 CUDA 实现，明确不是旧 PIL 增广的逐位复现。current 与 replay 使用独立显式随机事件键，current 顺序和增广在同 seed/task/epoch/batch 的所有分支、方法间严格配对。正式运行前做 train-only 吞吐和配对检查。
- 训练采用 channels-last 内存布局、BF16 autocast，master parameters 和 SGD 状态保留 FP32，无 GradScaler；测试推断禁用 autocast，使用 FP32 logits。启用 deterministic algorithms、关闭 TF32，锁定 `.venv-r9-locked` 环境。保存实际软件版本、GPU、源码哈希、初始化/前后 checkpoint、每epoch损失/更新数/样本数和顺序身份。该混合精度训练不声称与旧 FP32 轨迹逐位等同。

## 辅助操作与终点

固定晚期四任务 t=6,7,8,9。每个 pre-state 上施加70轮 current-only SGD probe，与 actual 使用相同 current 次序、增广、优化器和学习率日程。Fresh 从该 seed 的完整初始网络运行相同 current-only 操作。SGD actual 与 probe 的算法及起点完全相同，因此允许明确的内容寻址复用；不能把复用称为新增训练。Fresh 可跨方法复用，但需配对起点/随机事件和最终参数哈希证明。全部有实际执行或显式 identity 的来源。

每个终点保存全部10,000张官方test的100维 logits、原始/映射标签、原始索引、类映射和checkpoint哈希。推断不改权重。主 all-seen 预测只用当时已见输出，current-ten mask只用于辅助候选诊断并明确供应task identity。未见未来类图片不进入该任务的current/old指标，不作为独立任务重复。

同一个终点同时报告 current、pooled old、各old-task和seen-all accuracy，不拿actual的保留率代替probe保留率。没有正保留率匹配或部署效用主张；probe若遗忘旧任务，直接报告。

## 冻结估计量、判败和统计

主单位是配对 seed，先在固定 t=6..9 内等权平均，再做五seed汇总。该估计量不同于 CIFAR-10 的 t=1..4，也不同于旧 CIFAR-100 的单末任务。

对每seed计算三个 pp 差：

1. `d_actual = mean_t(actual_ER_allseen - actual_SGD_allseen)`；
2. `d_probe = mean_t(probe_ER_allseen - probe_SGD_allseen)`；
3. `d_masked = mean_t(actual_ER_current10 - actual_SGD_current10)`。

待检验的原联合方向是 `d_actual<0, d_probe>0, d_masked>0`。严格复制门要求五seed均满足全部三个不等式；零差不算成功，缺失/失败seed不能删除或用替代seed补足。这相当于五个独立seed的联合方向指示均为1；在条件于固定数据/实现的成功概率不超过1/2原假设下，精确单侧二项尾概率为1/32。此概率不是三个均值的显著性检验或接收概率。

方向由原始整数 correct/n 的有理数差判定，避免浮点抵消把精确零算成成功。每任务恰有1,000张current test图，四任务平均pp差等于四项正确数差总和除以40。

完整报告每seed三个差、三个样本均值/SD、每个单独方向及联合方向的计数；五seed配对均值区间采用全部 `5^5=3125` 个有放回重采样的2.5/97.5百分位，仅作为小样本近似bootstrap区间，不能宣称精确95%覆盖。保留率、fresh-relative差、各task分解及训练损失为预设描述性支持结果，不替代主门。

若任一方向在均值上不成立，则联合现象不推广到此设置；若均值方向成立但五seed门失败，结论为有限/不稳定推广。若主门通过，仅支持正常预算下的这组经验联合比较；不能据此识别内在plasticity、证明probe部署更优或宣布新机制/strong accept。阴性结果同样进入稿件。

## 执行与资源

用户明确实验运行时间不计入本轮一小时主动工作预算，并要求省token。先做不读test的资源witness，验证最坏ER batch及辅助操作，再冻结完整manifest后执行。优先每个seed/method作业不超过一小时；若实际吞吐不满足，先优化工程或完整说明预计时长，不以缩水训练轮数救出目标结果。每个作业有明确硬时限，超时是未完成，不能拿提前checkpoint替代固定终点。整组运行保留进程日志和完成/失败状态，不根据中途accuracy调整矩阵。

冻结前 train-only 工程比较支持采用 BF16：保持增广配对、task0方法身份及有限损失/参数，完整矩阵投影相对 channels-last FP32 约加速1.39倍，SGD/ER作业约29.5/39分钟，五pair训练约5.72小时。原拟以至少1.5倍加速作为采用条件，实测未达到；在看到吞吐后明确修订工程采用条件，理由是1.39倍仍降低总时间并使ER单作业明显低于一小时。保留此次工程决策变更，不回称满足原1.5倍条件。该选择未访问test或按accuracy选择，科学矩阵和判决门不变。设置每作业5,400秒硬保护以容忍热降频；它不是预计耗时。实测记录在 `audit/r9/one_hour_push/r15_witness_v3_bfloat16/RESOURCE.json`，最终精度/布局以冻结manifest为准。

正式结果后由独立新上下文审查者核对raw、分析与可宣称结论；重新生成论文段落/表，编译，检查变化页及全稿页数，封装新独立companion并核对其统计复算。旧V11完整包保留，不冒称新实验已在旧包内。
