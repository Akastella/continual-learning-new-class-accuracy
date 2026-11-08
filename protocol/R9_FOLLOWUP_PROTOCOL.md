# R9 决定性补充实验协议

2026-09-09。冻结以 `configs/r9_followup/manifest.json` 的UTC、协议/源码/配置SHA256为准。先前草案不是冻结计划。此实验由已公开的36-run发现结果选择问题；只确认新的运行，不能将发现集重称确认集。

## 核心问题与边界

检验官方CIFAR-10中，ER相对SGD的实际适应gap与统一probe gap是否给出相反排序，以及额外训练样本和当前梯度系数能否解释它。主设置为class-incremental、4epoch适应终点。A/H/C是操作定义，A=H+C不是理论发现；H不是模型固有能力。不主张领域中所有plasticity定义错误。

## 固定40条历史

- 两种协议：classIL（10输出、训练及评估均无task mask）和task-aware（同10输出参数、逐样本两类mask）。
- 两种历史算法：SGD、ER；新seed201..210全部运行并保留。每seed的初始化seed=seed+10000，batch顺序seed=seed+30000，类别顺序seed=seed+40000。方法/场景共享配对随机序列。
- 完整官方CIFAR-10：train50000/test10000。每seed对10个原类别做一次随机排列，依次两类组成5个任务，每任务train10000/test2000。训练和测试使用同一个全局类别ID双射，原始类别和映射全部记录。
- 每task4epoch，batch64，最后不足batch保留；无数据增广。CNN 32/64/128卷积、每层ReLU与2×池化、256维全连接、10输出；结构继承pilot。普通SGD学习率.01、无动量/weight decay。
- ER总reservoir500，每step先从过去呈现记录中不放回抽min(32,buffer长度)，之后将当前batch每次呈现加入reservoir。多epoch重复呈现照计。损失为当前和replay样本拼接后的mean CE。
- 不做test驱动调参。新种子独立于发现种子，但仍条件于同一官方数据；不宣称新外部测试集。完整训练规模、随机类别顺序是本批相对pilot预先声明的改动，不把差异仅归于seed数量。

## 同状态分支

每task前保存实际历史参数及memory。actual是唯一回写历史的分支；所有其他分支结束即丢弃，保留checkpoint及记录。fresh与各probe的当前batch顺序相同。

1. 标准fresh/probe：初始参数/任务前参数，current-only SGD，4epoch及0/1/2/3/4评估。
2. weighted_fresh/weighted_probe：在两种历史算法上均运行。current CE乘本step的nc/(nc+nr)，nr按该task时刻ER虚拟reservoir长度计算，只移除replay梯度；不从memory读取样本。任务前虚拟长度由之前完整历史的当前样本呈现数决定。
3. compute_fresh/compute_probe：在两种历史算法上均运行。把ER的nr个replay样本替换为从当前task训练集有放回抽取的nr个样本；固定独立随机流、配对初始化/顺序，损失为当前主batch与额外当前样本的拼接mean CE。匹配ER的optimizer steps与labeled forward examples，不声称匹配精确FLOPs或墙钟。额外样本单列extra_current_examples，不能记作真实replay。
4. compensated_er：仅在ER任务前状态上运行，复制memory及其随机状态，损失为CE_current + nr/(nc+nr)*CE_replay，恢复当前系数1而保留replay项绝对系数。它不替代实际ER历史；称敏感性控制，不称新算法。

所有系数按每batch实际大小计算。train10000、batch64的最后batch为16，满buffer时当前ER系数是1/3；普通满batch为2/3。不能将全程误写成固定2/3。

## 端点与推断

每seed先对task1..4等权平均。主端点为classIL、prefix4的配对向量(ΔA,ΔH)=(A_ER-A_SGD,H_ER-H_SGD)。预先定义的排序复现方向为ΔA>0且ΔH<0。

配对cluster bootstrap使用seed作为重采样单位，同一个100000×10索引矩阵同时重采样两端点，PCG64 seed=2026090901。每端点取1.25%和98.75%分位数（linear）；这是两项Bonferroni校正的percentile bootstrap区间，名义family覆盖95%，有限10seed的实际覆盖仅近似。原始seed值、样本均值、SD、完整区间均报告，不称精确分布无关保证。操作上的“复现排序”要求ΔA下界>0且ΔH上界<0；这不是论文接收判据。

次要分析全部公开：prefix1/2；task-aware；四个leave-one-task-out平均（仍在seed内）；方向性异号的margin0/.5/1/2/5pp全曲线；标准最终F/Acc及全部任务。它们不得替换主端点。除上述主比较外均为描述/敏感性，不以未校正区间补救失败主端点。

对两个current-only替代operator，fresh也用同operator，比较H_weighted_ER-H_weighted_SGD、H_compute_ER-H_compute_SGD；分别报告完整A/H/C及绝对fresh/probe精度，避免用fresh变难伪造改善。compensated_er使用标准fresh/probe构造A_compensated与C_compensated。额外样本和当前系数控制若解释了排序差异，就限制结论为具体操作条件，不把它当历史参数普遍损伤的证据。

## 失败、来源与下一步

不覆盖部分run或静默漏run。协议、所有源码、配置及数据/权重/训练计数冻结；缺产物/错hash/非有限数先调查修复，以新修订和新目录保留失败证据。完整矩阵后才进行组分析。

独立审查与工程smoke通过后启动。预计基础及控制在单RTX4060上需约1–2小时，实际记录为准。无新人工标注。若主现象未复现，则停止围绕该现象堆算法，重新评估科学主线；若复现且有可解释的诊断增量，下一阶段需要外部无矛盾数据集、现代plasticity控制及独立论文盲审。当前协议本身不保证达到用户的论文目标。
