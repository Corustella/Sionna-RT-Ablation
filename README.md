# Sionna RT Lunar Propagation Ablation

在相同地形、TX、材料、频率、射线预算和配对采样 seed 下，关闭传播机制，测量其对接收功率、覆盖与路径损耗的边际影响。**默认只研究 band 58，实际频率5.775 GHz。** 无DNN训练，不改动 Lunar RadioMap 预测仓库。

固定 `sionna-rt==1.1.0`。场景根据 LunarRM 的 `metadata.json` 和高度图重建；原始场景生成脚本、三角化及后处理细节未取得，因此本仓库不是官方标签生成器的精确复现。

## 六组配对配置

| 名称 | LOS | 镜面反射 | 漫散射 | 透射（refraction） |
|---|---:|---:|---:|---:|
| full | 1 | 1 | 1 | 1 |
| los_only | 1 | 0 | 0 | 0 |
| no_los | 0 | 1 | 1 | 1 |
| no_specular | 1 | 0 | 1 | 1 |
| no_diffuse | 1 | 1 | 0 | 1 |
| no_refraction | 1 | 1 | 1 | 0 |

只修改求解器开关，不将材料的 scattering coefficient 改为0。该版本没有 `diffraction` 参数；不添加新版API的 `diffraction=False`。`refraction` 是单表面附着平板的透射模型，不是实际山体内部路径长度模型。

## 服务器环境与同步

建议使用独立环境，避免影响现有训练环境。以下环境名为本仓库新建环境，不是假定服务器已有同名环境。

```bash
cd ~/liuanda
git clone https://github.com/Corustella/Sionna-RT-Ablation.git
cd Sionna-RT-Ablation
conda create -n lunar-rt-ablation python=3.11 -y
conda activate lunar-rt-ablation
python -m pip install -r requirements.txt
```

后续同步：

```bash
cd ~/liuanda/Sionna-RT-Ablation
git status
git switch main
git pull --ff-only
conda activate lunar-rt-ablation
```

需要兼容的NVIDIA GPU/驱动。默认显式使用CUDA，不静默退回CPU；`--backend llvm` 可用于有LLVM支持的CPU环境。Mitsuba/DrJit由Sionna RT 1.1.0依赖锁定为3.6.2/1.0.3；不要随意升级求解器。

先检查统计、几何及实际RT接口：

```bash
python -m unittest discover -s tests -v
python tests/smoke_rt.py --backend cuda
```

后者只运行8×8合成平面、深度3、每配置每seed约两万条射线，并验证分批发射方向和LOS/镜面路径功率与单次求解一致；不是正式实验或精度验证。

## 5.8 GHz首轮运行

默认数据路径示例为 `~/liuanda/LunarRM`；不复制、不修改数据。支持 `<split>_<field>.npy` 及 `<split>/<field>.npy` 合并数组布局。高度图按 `hm_row`，目标与mask按index行读取；mask解包，不根据145 dB判断有效性。

预检查只读输入，不导入RT、不创建输出目录：

```bash
python run_ablation.py \
  --data-root ~/liuanda/LunarRM --split val \
  --sample-ids 00016_00 --bands 58 --seeds 0 1 \
  --samples-per-tx 100000 --rays-per-batch 100000 \
  --out runs/smoke58 --dry-run
```

真实地形接口试跑（六组×两个seed=12次配置运行）：

```bash
python run_ablation.py \
  --data-root ~/liuanda/LunarRM --split val \
  --sample-ids 00016_00 --bands 58 --seeds 0 1 \
  --samples-per-tx 100000 --rays-per-batch 100000 \
  --out runs/smoke58
python analyze_ablation.py --run runs/smoke58
```

**十万条射线仅用于接口和耗时检查，不能据此确定机制贡献。** 正式小规模探索建议先选8个terrain，每地形1个TX，每配置每seed一千万条射线：

```bash
python run_ablation.py \
  --data-root ~/liuanda/LunarRM --split val \
  --terrains 8 --tx-per-terrain 1 --selection-seed 20261026 \
  --bands 58 --seeds 0 1 \
  --samples-per-tx 10000000 --rays-per-batch 1000000 \
  --out runs/pilot58_10m
python analyze_ablation.py --run runs/pilot58_10m
```

这是96次配置运行、合计9.6亿条发射射线。官方metadata为**每TX每次4×10⁹**，因此一千万仍是较低采样预算的探索，不是已收敛结果。先根据试跑实测时间估算总耗时；不能提前承诺分钟数。先看两个seed的差异、零命中覆盖和LOS闭合误差，再决定是否在相同样本上提高预算。只有在结论对采样预算稳定后才解释贡献大小。

增加预算时创建新 `--out`；同一预算中断时原命令追加 `--resume`。分批检查点记录完成的射线数及加权和，已完成配置跳过；更改代码版本、数据标识、依赖、几何或实验参数会拒绝恢复。正式运行和恢复应使用已提交的干净工作区；未提交代码不能验证恢复一致性。不要同时让多个进程写同一输出目录。分析目录已存在时，用新的 `--out`，不覆盖旧报告。

默认最大交互深度取metadata的100，禁用Russian roulette，不使用gain停止阈值；原始停止规则未公开，这些约定记录在manifest。`--max-depth` 可以显式改变，但结果属于另一实验条件。415 MHz仅在明确指定 `--bands 415` 或 `--bands 415 58` 时运行。

## 输出与贡献解释

`runs/`、`results/`、`docs/`均被Git忽略，服务器分析不依赖docs。

- `manifest.json`：代码版本、数据路径/大小/mtime、完整metadata、样本行、全部开关、预算、实际batch seeds、环境及命令。
- `inputs/`：本次选中输入、官方mask/标签、三角形面积/重心和固定插值覆盖域。
- `maps/<sample>/<band>/seed*/`：每组的原始三角面元线性path gain、投影后的线性地图、样本数和耗时；同目录保存分批进度。
- `analysis/summary.csv` / `summary.json`：主要统计和逐seed结果。
- `analysis/terrain_comparisons.json`：按terrain合并TX后的比较；另报等权terrain平均，避免多TX地形被当成独立地形。
- `analysis/comparisons.json` / `label_diagnostics.json`：逐样本、seed、机制的详细统计。

主机制统计直接使用原始接收面元，线性功率比例按面积加权；另报官方有效像素与固定插值覆盖域的交集。dB分布为共同正功率面元/像素的等权统计。对于去除机制X：

\[
\eta_X=\frac{\sum_i w_i(G_{full,i}-G_{noX,i})}{\sum_i w_iG_{full,i}}.
\]

它衡量被删除路径族的**边际功率影响**。镜面+漫散射等混合路径会被多个开关同时删除，所以三个比例不能相加当作互斥能量分解。它也不等于DNN可改善的RMSE比例。负值保留，不裁剪成0；负值/反常增强应先检查Monte Carlo噪声与数值稳定性。

同时报告：

1. 变为零的像素/面元数及占原正功率区域比例。
2. 两组都为正功率处的 `ΔPL=10log10(G_full/G_noX)`、分母及分位数。
3. 原来为零而消融后非零的数量，以及消融后变强的数量。
4. 两个seed分别的结果及先在线性域平均后得到的结果。
5. `full−los_only−no_los` 的有符号和绝对闭合误差，作为数值诊断。

零值表示有限采样下零估计，不能自动认定物理上绝无路径。对每组使用自己的positive mask再单独求平均会隐去覆盖损失，本实现不会这样计算线性贡献。dB差只在共同正功率域计算，并明确给出覆盖损失；不以任意floor制造有限dB结果。

`label_diagnostics`仅在“官方有效、插值可覆盖、模拟功率为正”的交集报告RMSE，并同时报告缺失数量。**这不是完整官方masked RMSE，也不能跨配置忽略分母差异直接比较。** 重点先查看full与标签的偏差，理解重建场景的适用范围。

早期版本（含 `cf5d143`）的重心插值可能保留本应为零的微小浮点权重，使少数零命中像素产生假正功率和极端dB值。当前将小于 `1e-12` 的无量纲重心权重归零并重新归一；这不是功率阈值，不删除真实弱信号。已有产物无需重跑RT，可从保存的面元功率重投影，原文件保持不变：

```bash
python analyze_ablation.py --run runs/pilot58_10m \
  --reproject --out results/pilot58_10m_reprojected
```

## 几何、聚合及适用边界

- 1 m格点的 `x=column, y=row, z=height`，每格固定NW–SE对角线，法向由三角形决定，不进行顶点法向平滑；也支持显式NE–SW对角线对照。
- 地形为开放高度场，无凭空增加的侧墙或底面；TX在本地地表+3 m，测量面为相同mesh整体上移+1 m（具体数值读metadata）。
- 直接在数组行列坐标中建模和输出，不额外flipud。不能把这个约定当成已经复原官方的原始坐标变换。
- **该版本初始发射采用固定Fibonacci格点，seed仅影响后续交互抽样。** 重复调用小预算solver会重复初始方向，不能当成更密的角度采样。本实现对同一套全局 `samples_per_tx` 方向按索引分片，禁止把重复的小格点当成完整大格点。
- v1.1.0没有公共分片接口；适配器仅在单次调用期间替换发射格点函数，结束或异常时恢复，不修改已安装源码。传输/材料方程不变；不能在同一进程并发调用该适配器。射线初始方向与同总预算原求解器一致，分批交互随机流不保证与单次求解逐位一致。
- 原始面元功率按射线数加权合并，将每batch的 `4π/N_batch` 归一换算为全局的 `4π/N_total`；各batch和seed的零估计都保留。Russian roulette与gain阈值关闭，避免依赖幅度的截断破坏这项缩放关系。没有随缺测程度改变预算，没有只平均命中的样本，也没有最终补洞。
- 重心到网格使用固定三角剖分的线性**功率**插值，不能跨机制改变插值点集。重心凸包外为NaN，单独报告；官方的插值/跨seed平均计算域仍未确认。
- 不同路径非相干累加；改变机制会改变抽样概率，但求解器有重要性抽样权重补偿。保持材料参数不变，仍须检查有限采样误差。

求解器时间主要随射线数、实际交互深度和场景求交成本变化，不能由DNN推理500 ms限制推算。该仓库用于**离线机制研究**。每batch工作量受 `--rays-per-batch` 控制；累计功率数组空间O(面元数)。三角网格O(HW)，一次Delaunay插值模板约O(M log M)，后续每图投影O(HW)。分析主要为线性扫描及分位数计算O(N log N)。

## 一手实现依据

- [RadioMapSolver v1.1.0](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_map_solvers/radio_map_solver.py)：四个机制开关、接收面、采样和停止规则。
- [MeshRadioMap v1.1.0](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_map_solvers/mesh_radio_map.py)：面元面积归一与路径功率累加。
- [RadioMaterial v1.1.0](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_materials/radio_material.py)：材料/极化、平板透射、事件概率与重要性采样补偿。
- [Fibonacci发射格点](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/utils/ray_tracing.py)：固定初始方向；本仓库仅按全局索引分片计算同一数学序列。
- [挑战官方参考代码](https://github.com/anderspearson206/lunar-radiomap-challenge) 与本地数据包metadata：频段、材料、TX/RX等参数。

文献/实现支持上述物理和数值定义；本项目贡献程度必须由实际运行给出。当前代码不预设镜面、漫散射、透射的排序，不把试跑结果写成正式结论。

## 同次追踪的传播分量数据集

新增的528地形、5.775GHz、每seed40亿射线数据生成与路径诊断见
[DATASET.md](DATASET.md)。正式数据放 `dataset/`，额外消融/复算放 `runs/`、
汇总放 `results/`。原有六组消融入口保持可用。
