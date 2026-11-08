# R9：独立数据上的正保留 checkpoint 选择

## 问题和贡献边界

已完成的 CIFAR-10 follow-up 在固定 4-epoch current-only 探针终点得到所有旧类正确数为零。因此不能把实际 ER 的旧类准确率与探针的当前类准确率合成一个模型。本实验只检验：在相同验证旧准确率要求下，两个冻结历史产生的、按相同规则选出的适应 checkpoint 在新数据上表现如何。

这是后续描述性外部诊断，不是新算法、首次提出 continual evaluation、在线 early stopping、分布保证或原主假设的追加确认。问题和阈值受旧实验发现影响；CIFAR-10.1 未参与原历史训练。固定新数据不消除整个研究的事后选题。原 SGD/ER 配方与短 CNN 预算的局限保留。

## 完整矩阵与预算

- 仅原 class-IL seeds 201--210、SGD/ER 共20历史，final task t=4，不根据结果增删 seed/task。
- 每条从原 pre_task_004 参数开始，严格复用原 current-only SGD：lr .01、momentum 0、weight decay 0、batch64、原类映射/归一化/顺序，4epochs、每epoch157步，共628步。
- 每条分支不延长原4epoch预算；重新执行轨迹构成新增计算。旧作者候选中1252步来自错误的batch32估计，不适用于冻结原实验。
- 固定候选步：[0,1,2,4,8,16,32,64,128,157,314,471,628]，13个。最终state必须与原probe_task_004全部tensor完全一致。actual post_task_004 是独立锚点，不参与探针选择。
- 全矩阵12560optimizer updates；计算开销包含全部628步和所有候选验证，不按获选步报告成本。整组55分钟 watchdog，无按结果削减。
- 先执行完全不读test的工程资源见证；无完整管线时间证据不启动科学矩阵。

## 新数据、唯一划分、访问顺序

作者官方 CIFAR-10.1 v6 commit d9982abb0bfc4846b8d13a11e66b887d946205d0；图片SHA256 2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6，标签SHA256 ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448。字节获取时未解码或运行模型。

原始类别c的200个行索引，按PCG64(seed=20260915+c)独立置乱，前100 validation，后100 test，各集合最终按官方行号排序。全项目所有模型共享同一个划分，不随训练seed重划。每个集合1000图，每类100图；对于最终任务有800旧图、200当前图。预处理与原训练完全一致；全10输出候选。仅使用作者公开标签，无新人工标注。

模型访问前只解码公开标签，生成并保存 `configs/r9_retention_trajectory/split.json` 的明确索引，SHA256 `925db2e56165374a7b3a3855e9934bc6f2818fb07433a3404c88911f10a6f31e`。该文件与生成器一起纳入最终冻结；运行时重新计算并逐项比对。

先完成全部20条轨迹的validation推理与选择，将全局selection.json封存后才运行任何test forward。加载原始array及标签以执行预定划分不属于模型测量；首次test forward时间另记，必须晚于全局selection seal。test只评价已选checkpoint及三个固定锚点pre/terminal/actual，不对所有候选绘制test oracle曲线。以后不据test改变阈值/网格/划分。

验证使用旧类标签，属于评估资源；不能描述为零旧数据访问的SGD部署，或训练内memory-free政策。原始数据不随复现包重新分发，保留官方来源/版本/校验和下载步骤。MIT仅适用代码，不适用图像/标签。

## 固定选择规则与报告

主要求q=20%旧类准确率，q=10%敏感性要求，q=0%无约束参照。10%恰等于十候选均匀随机准确率，不能称作有用保留；20%平均也不保证每个旧任务保留。必须报告全部十类整数正确数、四个旧任务准确率。

每历史、每q在13候选中，先用验证整数条件100*old_correct >= q*800筛选，再最大化validation current_correct；并列取最小step。空集记infeasible/NA，绝不回退到不满足条件的分支。actual不是候选。选定step后只取该checkpoint的test old/current，test旧准确率低于q记violation，不重选/剔除。

约束只适用于所选终点；不保证整个适应过程或未测网格之间均满足旧准确率要求。

全量展示20历史的选择状态、step、validation与test旧/当前/各旧任务整数counts，给出每方法val可行数、test违约数。ER--SGD当前准确率差仅在两者validation均可行的预定配对seed集合计算，明确条件总体及n；保留所有test违约模型。不因test是否达标过滤差值。不把一个方法不可行时的条件均值当全10seed优越性。固定pre/actual/terminal锚点单独给出全10seed均值SD。

不作p值、确认性CI或“强接收”判断。报告种子配对差与描述性均值SD；训练seed同时改变初始化/类序，所有seed共享新图，因此跨seed离散度不能代替新图总体泛化不确定性。本实验只覆盖已冻结历史、该数据划分及有限13候选；infeasible不证明其他优化器/预算不可达。

## 完整性与停止规则

首次模型读取CIFAR-10.1前冻结本协议、源码、统计审计、tests、全部输入hash及机器UTC。固定原source/config/data_manifest/pre/probe/actual和官方训练数据hash。保存全部候选checkpoint、验证logits/labels/官方行号、选择seal、所需test logits、batch顺序hash、训练counts、环境、耗时、完整文件hash清单。

独立NumPy重建映射/划分/所有counts/选择/配对汇总；另以函数式CNN对seed201两方法全部实际test评分checkpoint独立forward。原终点state逐tensor一致是20轨迹的必要条件。超时、非有限值、源漂移、缺失seed、state不一致均记失败，不用部分结果替换完整矩阵。
