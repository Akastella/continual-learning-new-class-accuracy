# R13：早期网格与 actual 候选资格的固定政策检验

## 问题与来源

本协议针对当前 PDF 审查 Major 2 的具体缺口：旧 current-only 轨迹只在十三个步骤选择端点，可能遗漏最早期的可行模型；actual 只作为事后锚点，未经过同一验证选择规则。唯一科学问题是：**在同一历史状态与相同验证保留要求下，补齐前十六步或允许 actual 参与选择，是否改变所选模型及其在新图像上的表现？**

这是一项受已知 R9/CIFAR-10.1 结果启发的新政策敏感性检验，不修改或追认旧冻结协议。不是原主均值命题的独立确认，不提出新算法，也不声称解决 ER/SGD 起始状态差异或识别内禀可塑性。旧 CIFAR-10.1 validation 已被使用；旧 test 不用于本轮选择或新测试。测试采用作者官方 CIFAR-10.2 test，首次模型前向必须晚于本轮全局选择封存。来源、数组和重复图像检查均在前向前固定。

## 完整矩阵与唯一候选规则

- 原 class-incremental SGD/ER、seeds 201–210、final task=4，全部二十条历史，不按结果增删。
- sparse 步骤为原 `[0,1,2,4,8,16,32,64,128,157,314,471,628]`；dense 为其与整数 `[0,16]` 的并集，共24个。
- 四个政策：`sparse_probe`、`dense_probe`、`sparse_plus_actual`、`dense_plus_actual`。后两者只在相应 probe 集合加入本历史原 actual post-task checkpoint。
- 两个固定约束：`q20` 要求 validation pooled-old accuracy≥20%；`q20_rel5` 同时要求≥20%以及相对本历史 step0 的 pooled-old accuracy 下降≤5个百分点。5pp为本次事前固定敏感性约束，不调参、不依测试挑阈值。
- 在可行集合内最大化 validation current 正确数；并列先取较小更新步骤；同为628步时 probe 在 actual 之前。actual 的 tie rank=(628,1)，probe=(step,0)。空集为 null，不回退。
- 可行性用整数交叉乘法，禁止浮点 epsilon。相对约束控制每历史允许下降，**不匹配起始 old、最终 old 或历史参数，也不要求每个旧任务都达到阈值**。

## 历史与训练闭环

原 R9 follow-up 与 trajectory manifest、全部使用的源码/配置/pre/actual、旧13候选 checkpoint/validation logits 及训练数据均纳入本轮输入 hash。

只从各原 pre-task 状态重演前16次原 current-only SGD 更新，总320更新，batch64、原数据顺序、学习率、FP32及确定性配置不变。保存0..16全部17个状态；在0、1、2、4、8、16与旧 checkpoint 的每个 tensor 严格相等。原十三候选的验证 logits 直接复用；新增11步/历史才作新验证推理，另对20个 actual 状态补验证推理。新早期轨迹和旧16..628链以step16逐tensor相等连接；这不声称本轮重演全部628步。旧终点与原probe的既有严格一致性证据仍可追溯。

评估不能改变参数或训练随机数状态。新runner调用原冻结训练core的stop_step=16，不改旧resume identity或旧源码。任何源漂移、非有限值、旧节点不一致、缺seed、超过预算均失败，不将部分矩阵作为完整结果。

## 数据与首次模型访问

validation固定使用原CIFAR-10.1 v6 split.json中的1000张验证图；按每历史冻结class permutation映射标签，all10候选。不得使用旧1000张test的logits或结果选择本轮模型。

CIFAR-10.2仅使用作者官方test文件，commit、URL、完整SHA256与结构记录在 `third_party/r13_cifar102_reference/SOURCE.json`。先做数据结构、类别计数及与原CIFAR-10 train/test、10.1的逐图精确字节重复检查，不运行模型。若存在完全重复图像，不能称全体图像全新；完整报告且不得看结果后删样本。未知近重复不冒充已排除。原数组不打包分发，提供固定来源和核验脚本。

全部二十历史、八个政策/约束组合选择完毕后，保存 validation raw、选择 JSON 及全局 SHA256 seal，随后才允许任何10.2模型前向。测试仅评分被至少一格选中的去重 checkpoint，以及固定 pre/actual/terminal 三锚点；不生成所有候选的test oracle曲线，不测试后重选。

固定20历史的所有选择都保留。报告每个selected模型的old/current/all及四旧task整数counts；测试绝对违约以test old≥20%计算，相对违约以同一历史test pre到selected的下降≤5pp计算。不能用test可行性删点。分母取实际官方标签计数，不硬编码10.1的800/200。

## 预定分析与失败解释

逐seed展示八格选择/NA、validation和test整数counts、相对pre变化、测试违约、实际anchor胜负。主要描述量为同一方法/约束内 dense_probe−sparse_probe 的配对current/old差；另报告加actual的同历史政策差，以及四格间所有可用性变化。仅在两个被比较政策validation均可行时算配对均值和样本SD，明确共同subset，不相减不同分母的单组均值。

ER−SGD的同政策差是条件于共同可行seed的次要描述；必须并列pre状态与实际锚点，禁止将其解释为初态匹配后的适应能力。所有十seed逐项列出，整数差先转Fraction再做聚合/符号；无p值、确认性CI或测试后成功阈值。

若dense不改善或选点相同，限制只适用于本前16步补齐，而不是全轨迹不存在其他可行点；若dense改变结论，旧十三点诊断的分辨率敏感性须显式进入论文；若actual赢得选择，说明原probe候选资格有实质影响，不是新策略击败actual；若增加验证候选使test更差或违约增加，照实报告选择泛化失败。无论方向如何，不把grid密度或更多数据集本身当新颖性。

## 成本与验收

整组硬watchdog=3300秒，包含加载/输入验证、320更新、240个新增validation端点、所选test推理和raw落盘。旧完整20×628组真实耗时115.25秒仅是资源见证，不是本轮实测时间；本轮记录独立实测。最多每历史4个不同probe赢家加3锚点，≤140个测试端点；该界依赖四政策包含关系和固定tie规则，分析必须核对。

启动前冻结本协议、实现、测试、分析、固定data/source/旧evidence hashes及UTC。分析独立从raw logits/labels重算全部整数counts、选择、违约、配对与精确零；独立functional前向至少覆盖seed201两方法所有实际test评分状态及全部新validation状态。完成后进行fresh same-family完整性及result-to-claim核验，不给接收分；实际运行证据不等于科学主张通过。不得声称论文原Major 1/4或强接收目标已由本实验关闭。
