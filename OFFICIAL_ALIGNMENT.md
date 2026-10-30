# 官方已知口径的核验与标签导出

在 `research/component-dataset` 的 FP64 修复上增加独立后处理，不修改 RT、
地形剖分、插值模板、分量归约、生成计划或断点。原始分量图保留。
这只是**部分对齐**，不是原始官方生成器的复现认证。

依据是数据包 `metadata.json` 与
[Sionna RT 1.1.0 原实现](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_map_solvers/radio_map_solver.py)。
官方论文描述了重复 seed 补缺失与最终填洞，但没有提供完整生成脚本：
[RadioLunaDiff，§4.2](https://arxiv.org/html/2509.14559v1#S4.SS2)。

## 本轮处理边界

- 核验 manifest 中的 5.775 GHz、80 MHz、RT 1.1.0、每 seed 40 亿、深度100、
  full 四机制/无绕射、材料参数、TX 3 m/20 dBm、RX测量面+1 m等声明。
  当前生成代码已有这些设置，因此不再改动它们。
- 核验保存的测量顶点确实是地形 mesh 的 +1 m 副本，面序不变，TX遵循当前
  `(col,row,H+3)` 约定。这不证明当前世界坐标约定等于原始官方约定。
- 保存的规则网格先保留当前线性增益插值，再按像素忽略未解析的 seed，
  用有限值平均；最后未解析处填145 dB，输出正路径损耗 float16。
- `--mean-domain` **必须显式指定**：metadata 只写 `np.nanmean`，未公开其值域。
  `gain` 保留本项目原有的线性增益域假设；支持 `db` 以便以后有原始依据时使用。
  本轮不依据标签误差选择值域，也不修改原插值。
- 不实现未知的收益递减截止规则；不声称固定 seed 0/1 代替了官方最多15次
  初始尝试、从seed30开始最多60次精化。首次/精化 NaN 阈值的计数时机、原seed起点、
  平均值域和收益递减判据仍需要原脚本依据。保留未知项，避免编造默认值。
- 不盲加 `flipud`：metadata 写有该步骤，但 mesh、TX、query 的完整坐标上下文未知。
  也不改变原网格/法向、不新增绕射、不继续搜索更低RMSE的几何或插值假设。

严格说，已确认的是有限值平均这个操作；**在线性增益域如何表示缺失和执行它**
仍是保留现有值域的显式实现假设。结果 manifest 会记录它，不能称为已知官方均值。
补145及float16只对齐最终表示，不能恢复缺失路径。

## 输出和比较

每场景要求所有指定 seed 均有完整40亿地图，否则记录缺失并跳过，绝不把进度断点
当成完成地图，也不悄悄改成单seed输出。已有导出目录拒绝覆盖。

```text
dataset/aligned_labels/
  manifest.json
  <partition>/<sample_id>.npz
    pathloss_db             # float16，未解析处145
    resolved_mask           # 本次重建是否有有限PL；不是官方评分mask
    unfilled_pathloss_db    # float64，未解析仍为NaN
    contributing_seeds     # 每像素有有限值的seed数
    requested_seeds / mean_domain
results/official_known_alignment/
  maps.csv / pooled.csv / summary.json
```

对照包括各单seed、相同seed数且零增益参与的旧线性增益平均、有限值平均导出。
所有对照都填145、转换float16，并在**同一官方有效mask的全部像素**上评价。
主指标为 pooled SSE/count 后开方；另外报告覆盖率和填洞前的重叠域RMSE。
`resolved_mask` 不得用于缩小主评分分母；145也可能是真实有效值，不能按值推断mask。
增加seed的收益与更改缺失处理的收益需分开解释。

算法时间 O(SHW)，工作内存 O(SHW)，S为所选seed数、H=W=256；
geometry核验另外逐场景读取 O(M) 顶点/面。它读取已有地图，不运行RT，无新增依赖。

## 服务器命令

先激活已确认可用的环境。以下只读已有地图，不续跑生成器。
新分支包含的生成代码仍是440ab38；生成器严格记录git版本，**不要在此分支对
原dataset调用 `generate_dataset.py --resume`**。原生成任务继续使用原生成分支。
若要与生成并行操作，建议单独克隆导出分支，避免切换正在使用的工作树。

```bash
cd ~/liuanda/Sionna-RT-Ablation
git status
git fetch origin
git switch research/official-known-alignment
git pull --ff-only

python -m unittest discover -s tests -v

python export_aligned_labels.py \
  --dataset ~/liuanda/Sionna-RT-Ablation/dataset \
  --out ~/liuanda/Sionna-RT-Ablation/dataset/aligned_labels \
  --results ~/liuanda/Sionna-RT-Ablation/results/official_known_alignment \
  --subset all --seeds 0 1 --mean-domain gain
```

`gain` 是保留原值域的假设，不是官方已确认值。需要复跑时换新的导出/统计目录。
生成任务的续跑继续切回 `research/component-dataset`，沿用 `DATASET.md` 的命令，
不得手改manifest绕过版本检查。本地测试与服务器执行是两件事。
