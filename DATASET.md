# 5.775 GHz 传播分量数据集

本分支新增同次 full 追踪的互斥分量导出，原 `run_ablation.py`、`rt_backend.py`
及旧六组消融保持原行为。正式产物默认写入仓库的 `dataset/`（Git 忽略），
服务器对应 `~/liuanda/Sionna-RT-Ablation/dataset`；其他预算复算和开关消融写入
`runs/`，统计报告写入 `results/`。

## 已固定的生成方案

- 528 个独立 terrain：官方 train 中 400 个拟合、另外 64 个内部验证，官方 val
  中 64 个外部验证；每地形随机选择一个**已有 TX**，先选 TX 再做几何分层。
- 每场景 seed 0、1 **分别**发射 4,000,000,000 条射线，max_depth=100，
  full 同时开启 LOS、镜面反射、漫散射、透射；RR、增益停止阈值关闭。
  1056 次 full，共 4,224,000,000,000 条射线，不为每个分量重复追踪。
- 16 个试点属于 400 个拟合集；4 个单机制控制属于这 16 个；代表子集 32 个
  （拟合 24／内部 4／外部 4），其中几何阴影更强的 8 个用于五 seed 检查。
- 地形分层：train 几何拟合起伏三分位、TX 相对高度中位数，结合粗 LOS、TX
  中心/边缘和单/多 TX 类型，按现有分层轮转抽样。粗 LOS 用 8×8 RX 和每条
  线段 33 个内部采样点，仅用于选择，不能当作精确 LOS 标签。
- 选择不读取 loss/mask。已在旧实验中检查的 8 个 val 地形排除在外部验证外。
  索引缺少 parent_id，记录 null；**没有保证父地形隔离**。外部 64 地形在模型
  冻结前不要用于拟合、选超参数或反复改特征。正式 PL 成绩仍用完整官方 val。
- `selection.csv` 和 manifest 固定样本；`--subset` / `--mode` 只改变本次执行范围，
  不改变该计划。复算通过 `--selection-from dataset/manifest.json` 复用划分。

射线预算、材料参数、频率和深度按 metadata 对齐；原始官方网格、坐标翻转的
具体上下文和自适应补采样脚本尚未取得。本数据是 **metadata 重建的组件标签**，
不宣称精确复现官方生成器。原生 mesh 结果不翻转、不填洞；附加规则网格沿用
当前项目的固定线性增益插值（不 flipud）。与官方标签的差异仅作重建诊断。

## 分量定义与实现

| 字段 | 完整路径历史（不计测量面穿越） |
| --- | --- |
| G_all | 全部原生线性 path_gain，无量纲 |
| G_LOS | 零次场景交互，不含透射 |
| G_spec | 至少一次镜面，且每次场景交互都是镜面，包括多次镜面 |
| G_diff_family | 有漫散射、无透射，可以含镜面 |
| G_trans_family | 有透射、无漫散射，可以含镜面 |
| G_diff_trans | 同时有漫散射与透射，可以含镜面 |
| G_other | 后三个分量相加 |

`component_solver.py` 继承 Sionna RT 1.1.0，保留其射线运输、电场、材料采样、
随机数调用和三角形面积/入射角/立体角归一化。仅新增每射线的交互历史位，
在命中测量面时把原有加权功率同时累加到总图和唯一的分量图。
历史位在场景交互之后更新，在下次测量面命中之前保留；测量面本身不增加历史。

检查 `G_all ≈ sum(五分量)`，每单元相对容差 5e-5，无增益绝对阈值；不裁剪
弱分量，不用独立消融差值替代标签。独立开关会改变随机采样/能量分配，其
边际效应不具有份额可加性。1.1.0 没有 diffraction 参数，代码也不传入该参数。
官方实现依据：[RadioMapSolver v1.1.0](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_map_solvers/radio_map_solver.py)、
[MeshRadioMap](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/radio_map_solvers/mesh_radio_map.py)。

40 亿射线按一个全局 Fibonacci 点集分批，默认每批 100 万，使用全局索引，
不是重复同一个较小点集。每批 seed=`SeedSequence([seed,batch_index])`。沿用上游
固定初始方向，因此 seed 只改变交互随机数，**不代表独立初始角度采样**。
每批内部求解累加为 FP32，按实际射线数在 CPU 以 FP64 加权平均（零命中批次
同样有权重）；输出 float64 不意味着追踪本身是 FP64。

复杂度：R 条射线、最大深度 D、M 个三角形、批大小 B，把单次场景求交视为一操作时，状态推进约 O(RD)，
分量新增常数次 scatter；地图/质量缓冲 O(M)，射线状态 O(B)，加上求交加速结构。
实际求交耗时取决于场景与加速结构。点路径诊断的历史缓冲约
O((samples_per_src + candidate_cap) × D)，另有公开系数张量的 RX padding；分 RX 批限制峰值内存。
不会一次分配 40 亿条射线或保存它们的全部轨迹。

## 目录与数组

```text
dataset/
  manifest.json                 # 参数、原始 metadata、数据文件身份、代码/环境/设备、命令
  selection.csv                 # 528 个地形/TX、source_split、hm_row、分层量、各子集标志
  geometry/<sample>.npz         # 真正使用的两个 mesh、顺序、中心、面积、H、TX、几何分组
  geometry/<sample>.json        # 对应关系、材料/天线、坐标定义
  reference/<sample>.npz        # 官方总损耗与官方有效 mask，仅用于重建诊断
  maps/<partition>/<sample>/seed0.npz
  maps/<partition>/<sample>/seed1.npz
  paths/<sample>/seed0/rx0000.npz  # 仅32个代表场景；seed1同样保留
  status.json                   # 已存在/预期文件数量，不代表采样收敛认证
  invocations/*.json            # 每次执行范围、耗时与命令
```

`maps` 每文件原生增益为上述七个 `(M,) float64` 数组，M=130050（256×256 地形）。
`hit_count`、`positive_hit_count` 是 `(5,M) uint64`，依次对应五分量：前者统计
测量命中次数，后者统计加权贡献严格大于零的命中次数；同一射线可以多次命中。
它们不是独立样本数，也不是方差估计。`unresolved_or_zero` 为七图的零值标志，
不能凭有限命中信息把零确认成物理真零。

`grid_gain`、`grid_pathloss_db` 为 `(7,256,256)`，顺序见 `gain_order`，共用一个
固定 XY 重心 Delaunay 模板，在**线性增益域**插值，原生结果仍保留。
`PL=-10log10(G)`；G=0 的损耗为 +inf，模板外为 NaN，绝不填 145 dB。
`interpolation_footprint` 独立于官方 mask。几何中 `centroid_los` 为实际 terrain mesh
线段求交（终点缩短 1e-4 m）；`region`=0 可见内部／1 LOS 面邻接边界两侧／2 阴影内部。
这是中心点可见性，不是整个三角形可见面积比例。

实际顶点为 FP32、faces 为 uint32；中心/面积由保存的实际顶点以 FP64 重算。
每个 scene 保存实际 TX xyz 和 orientation。全场景材料、TX 1×1 iso V、3m TX、
1m 测量面、20dBm 和 5.775GHz 等参数见 manifest，不改变 metadata 材料。

仅地图和几何未压缩约数十 GB，另有路径子集与临时 checkpoint；压缩大小和
实际路径数有关。不要按旧 1e8 消融 NPZ 大小估计完整交付体积。

## 点 RX 路径子集

32 场景 × 256 RX × seed 0/1，按实际中心点 LOS/边界/阴影和四个距离带轮转抽样。
RX 就在原生测量三角形中心，保存 `path_rx_cell_indices`。默认每批 32 个 RX，
`samples_per_src=100000`，最多 250000 个候选路径/源，max_depth=100。
**这是一组单独的 PathSolver 诊断参数，地图的 samples_per_tx 仍是40亿**。
先检查其候选数、覆盖与耗时，需要增加诊断预算时使用新输出目录并完整记录配置。
若候选缓冲触顶，在 shrink/无效路径剔除前报错，不接受可能被截断的交付。
需要改变路径参数时，可用 `--selection-from dataset/manifest.json --mode paths`
和新的 `--out dataset/path_diagnostics_v2` 单独导出，保留新 manifest；
这不会重新运行40亿射线的 full 地图。

每个 chunk 保存原生路径记录，去掉公开张量中人为的跨 RX padding：

- `a_real/a_imag[P,2,1]`：每条路径、RX 的 V/H、TX 的 V；1×1 共址阵列无额外
  合成阵元相位，保存 PathsBuffer 中的原始系数。不是接收功率归一化后的数量。
- `valid[P]`、`rx_index[P]`、`tx_index[P]`、`delay_s[P]`、角度 `[P]`；RX 索引是
  **chunk 内局部索引**，通过 `rx_cell_indices`、`rx_xyz` 回到测量面。
- `interactions[P,D]`、`vertices[P,D,3]`、`objects/primitives[P,D]` 保留完整深度；
  交互类型 NONE=0、SPECULAR=1、DIFFUSE=2、REFRACTION=4。零次交互是 LOS。
  尾部 NONE/invalid 位置不用作实际交互顶点。TX 和 RX 端点另存。
- object IDs 是本进程的指针编码，每个 chunk 保存 `object_runtime_ids` 与
  `object_mesh_names` 映射，面编号对应保存的 mesh；不能跨运行直接比较原始 object ID。
- 原始系数不合并极化；需要功率诊断时对 `a_real²+a_imag²` 的两个接收极化和
  路径做**非相干求和**。不先把复系数相加，不把点路径强行校准成三角形面积平均。

这些点路径不是 RadioMapSolver 的 Monte Carlo 射线日志，不能凭它们恢复
RadioMap 的加权功率。分量图已在 RadioMap 内使用原权重直接分组，和点诊断分开。
实现依据：[PathSolver v1.1.0](https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/path_solvers/path_solver.py)。

## 服务器同步和运行

先激活服务器上**已确认可用**的 Sionna 环境。依赖沿用 `requirements.txt`，
不需要新增包；环境还未安装时运行 `python -m pip install -r requirements.txt`。
以下由用户在 SSH 服务器执行，本地代码修改不代表已在服务器运行。

```bash
cd ~/liuanda/Sionna-RT-Ablation
git status
git fetch origin
git switch research/component-dataset
git pull --ff-only

python -m unittest discover -s tests -v
python tests/smoke_components.py \
  --metadata ~/liuanda/LunarRM/metadata.json \
  --backend cuda

python generate_dataset.py \
  --data-root ~/liuanda/LunarRM \
  --dry-run
```

先跑16个试点的正式40亿地图，输出就在最终 dataset；可以中断，原命令加
`--resume` 续跑。几何选择会扫描 train/val 高度图一次，后续从 manifest 读取。
默认每100批（1亿射线）保存一次 checkpoint，最多重算最后未落盘的100批。
正式跑前应根据首批实测耗时和预计输出大小确认计算/存储预算，不承诺固定小时数。

```bash
python generate_dataset.py \
  --data-root ~/liuanda/LunarRM \
  --out ~/liuanda/Sionna-RT-Ablation/dataset \
  --subset pilot --mode maps

python analyze_component_dataset.py \
  --dataset dataset --subset pilot \
  --out results/component_pilot_4b

python run_dataset_ablation.py \
  --dataset dataset --subset pilot --controls \
  --out runs/dataset_ablation_pilot_4b \
  --results results/dataset_ablation_pilot_4b
```

试点确认后，补齐528个场景地图，再生成32个场景的路径明细。两步共用同一个
manifest，模式/子集允许变化；其他物理/采样参数、批大小、代码、环境必须一致。
**不要在试点与全量之间切换代码提交**；若确有修复，应保留旧目录并新建输出。

```bash
python generate_dataset.py \
  --data-root ~/liuanda/LunarRM \
  --out ~/liuanda/Sionna-RT-Ablation/dataset \
  --subset all --mode maps --resume

python generate_dataset.py \
  --data-root ~/liuanda/LunarRM \
  --out ~/liuanda/Sionna-RT-Ablation/dataset \
  --subset representative --mode paths --resume

python analyze_component_dataset.py \
  --dataset dataset \
  --out results/component_dataset_4b
```

如需一次生成全部地图和路径，首次命令使用默认 `--subset all --mode all`。
分两步便于先检查地图及路径诊断的资源占用。无需创建新运行入口。

## 收敛复算与全量开关消融

32代表场景用相邻20亿/40亿预算比较；8难场景的40亿5seed复算也单独写入runs。
下列命令不改变 dataset 中的正式标签。五seed输出会包含0/1，可完整分析同一计划；
不会静默复用不同 manifest 下的部分文件。

```bash
python generate_dataset.py \
  --data-root ~/liuanda/LunarRM \
  --selection-from dataset/manifest.json \
  --out runs/component_convergence_2b \
  --subset representative --mode maps \
  --samples-per-tx 2000000000

python analyze_component_dataset.py \
  --dataset dataset --subset representative \
  --other-run runs/component_convergence_2b \
  --out results/component_convergence_2b_4b

python generate_dataset.py \
  --data-root ~/liuanda/LunarRM \
  --selection-from dataset/manifest.json \
  --out runs/component_hard_5seed_4b \
  --subset hard --mode maps \
  --seeds 0 1 2 3 4

python analyze_component_dataset.py \
  --dataset runs/component_hard_5seed_4b --subset hard \
  --out results/component_hard_5seed_4b
```

全528地形删机制消融默认包含 no_los、no_specular、no_diffuse、no_refraction，
每配置每seed同样40亿，直接使用 dataset 中 full 作基准，不重复生成 full。
额外4224次追踪、16.896万亿射线；只需镜面等三组时传
`--configs no_specular no_diffuse no_refraction`。先 `--dry-run` 检查工作量。

```bash
python run_dataset_ablation.py \
  --dataset dataset --subset all \
  --out runs/dataset_ablation_all_4b \
  --results results/dataset_ablation_all_4b \
  --dry-run

python run_dataset_ablation.py \
  --dataset dataset --subset all \
  --out runs/dataset_ablation_all_4b \
  --results results/dataset_ablation_all_4b
```

脚本直接输出逐图及 pooled RMSE 的 CSV/JSON：SSE 跨像素累加后除以数量开方，
不平均逐图 RMSE；比较域为同一 full/ablated 的正增益交集，并单列失去/新增的
零值支持。网格与原生三角形分别报告，三角形 RMSE 不按面积加权，功率积分按面积。
分量审计还包含 LOS/边界/阴影分组的 seed／预算 log功率误差、份额差和可靠命中
子集误差。可靠命中阈值默认8，仅用于诊断分组，不裁剪任何原始数据。
重建相对官方标签的 RMSE 同时报告 overlap 覆盖率，不当作比赛完整分数。

正式标签的训练用途仍是研究假设：低噪声、闭合标签可帮助学习几何到分量的映射，
是否提升最终 PL 网络必须再按地形折外构造特征、在固定完整 val 上做配对seed验证。
