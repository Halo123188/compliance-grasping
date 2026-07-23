# Stage 1 Toy Experiment: Free-Space Reach under Stubborn Human Disturbance

**给实现者的说明**：这是一份实现规格，不是研究提案。所有数字都是起始值，标注了哪些是可调的、哪些是结构性的（不能动）。凡是写"结构性"的地方，改了整个实验的结论就不成立了 —— 如果你觉得需要改，先回来问。

---

## 0. 这个实验存在的理由

这是一个三阶段计划的第一阶段：

| Stage | 任务 | 新引入的风险 |
|---|---|---|
| **1（本文档）** | Free space reach + 人为干扰 | 无接触物理。只验 arbitration + scaffolding |
| 2 | 加刚性物体接触 | 首次引入接触，但 rigid，sim 可信 |
| 3 | Grasping / handover / retry | 接触物理迁移、deformable、失败检测 |

**Stage 1 的唯一成败判据不是 reward 曲线，是：Stage 2 换任务时，env wrapper / obs 构造 / action 解码 / 网络 / 训练脚本 / 部署代码能原样复用多少。** 目标是除 task 和 reward 之外 100% 复用。

因此下面所有"看起来过度设计"的地方（各向异性 K、GRU、joint torque obs）都是故意的 —— 它们在 Stage 1 上可能没有收益，但它们是 Stage 2/3 必需的，现在建对比后面重写便宜。

**Downstream 场景**（决定了本文档的所有设计）：机器人抓取过程中被人干扰，要"一边让开一边继续抓"；或者直接从人手上接过物体。核心不是 compliance（解析律就能做），是 **arbitration**：task 和干扰在同一时刻要求相反的运动，policy 要决定怎么分配。

**Retry 走隐式**：不套高层状态机，靠长 horizon + 大位移干扰逼 policy 自己学出"重来"。Stage 1 里这只能做到最原始的 resume。

---

## 1. 非目标（明确不做）

- ❌ **不上真机。** Free space + 刚体动力学，sim 几乎精确，真机除了验证部署链路没有额外信息量。部署链路的验证并到 Stage 2 一次做完。
- ❌ **不做 teacher-student。** Stage 1 的隐状态只有 `K_h` + 推的方向 + 是否松手，维度很低，GRU 从头 RL 应该能学出来。真跑不出来再加（见 §8）。
- ❌ **不控姿态。** 只控平动，姿态用固定高刚度锁死。
- ❌ **不学 stiffness frame 朝向。** 只学 base frame 下的对角 K（见 §3.3 的已知缺口）。
- ❌ **不追求 tracking error 最小。** 那个一定会很小，且毫无信息量。

---

## 2. 场景

- **Sim**：MuJoCo。Flexiv Rizon 4S 模型（7-DoF）。能用现有的flexiv模型吗
- **任务**：末端到达随机 goal `x_g`，在 workspace 内 40×40×30 cm 的盒子里均匀采样（盒子位置需按 Rizon 的可达空间标定，实现时先跑 IK 可达性检查，采到不可达的点要重采）。
- **Episode**：5 s，policy 100 Hz → 500 步。成功即提前结束。
- **初始位形**：固定 home pose + 关节角小扰动 `U(-0.05, 0.05)` rad。

---

## 3. 控制栈

### 3.1 两层结构（结构性）

```
policy (100 Hz)  ──> Δx_ref (3), log K (3)
                          │  零阶保持 + 线性插值
                          ▼
analytical Cartesian impedance (1 kHz)
   F_cmd = K ⊙ (x_g + Δx_ref − x_ee) + D ⊙ (−ẋ_ee)
   τ_cmd = Jᵀ F_cmd + gravity_comp + nullspace_damping
```

- 姿态：固定 `K_rot = 50 Nm/rad`, `ζ=1.0`，锁在初始朝向。
- Nullspace：加小阻尼防止肘部漂移，不做别的。
- **Policy 永远不直接下发关节力矩。** 解析 impedance 层始终在，policy 只调它的参数。这是 fallback 能存在的前提。

### 3.2 Action space

Policy 输出 6 维，`tanh` 到 `[-1,1]` 后映射：

| 分量 | 维度 | 映射 |
|---|---|---|
| `Δx_ref` | 3 | 线性到 ±3 cm（硬 clip） |
| `log K` | 3 | 线性到 `log[50, 2000]` N/m，base frame 对角 |

`D = 2ζ√(K · m_eff)`，`ζ = 0.8` 固定，`m_eff = 2.0 kg` 固定标称值（**不要**用真实 Cartesian inertia，会引入构型相关的抖动，且 Stage 2 也用不上）。

> **不要让 policy 自由输出 D。** 自由学 D 很容易学出违反 passivity 的组合。这条是结构性的。

### 3.3 已知的表达力缺口（记录在案，Stage 2 再补）

各向异性只有 base-frame 轴对齐。Policy 没法表达"沿任意方向软"。Stage 1 够用（随机推的方向在统计上覆盖各轴），但 Stage 2 要加 stiffness frame 朝向，那时 action space 从 6 维变 9 维，`decode_action()` 要改。**把这个函数写成独立、易替换的模块。**

---

## 4. 人的模型（这一节是整个实验的心脏）

### 4.1 语义

人手是一个弹簧，`x_hand` 是人想让臂去的位置。人的意图是：**"你挪开，挪到那边去"** —— 一个位置目标，不是单纯的不抵抗。对应真实场景：人伸手进工作空间，把挡路的机械臂拨开。

### 4.2 状态机

```
idle ──(t = t_push)──> reaching ──> pushing ──(release cond)──> released
```

```python
t_push  ~ U(1, 3) s
attach  = link 7 (末端)          # Stage 1 只推末端，见 §4.5
u       ~ 随机单位向量 (3D)
d       ~ curriculum, 见 §6      # 目标位移
x_target = x_ee(t_push) + u * d

# reaching: x_hand 从 x_ee(t_push) 沿 min-jerk 走到 x_target
T_reach ~ U(0.3, 0.8) s
# pushing: x_hand 固定在 x_target 不动

F_raw = K_h (x_hand − x_ee) + D_h (ẋ_hand − ẋ_ee)
F_ext = F_raw * min(1, F_max / ‖F_raw‖)        # 饱和
K_h   ~ loguniform(100, 1000) N/m
D_h   = 2 * 0.7 * sqrt(K_h * 2.0)
F_max = 40 N
```

### 4.3 三个必须照做的点（结构性）

**(a) `x_hand` 要 ramp，不要瞬移，且力要饱和。**
瞬移 + `K_h=1000` + `d=0.2` → 瞬间 200 N。这既超出人的能力（人推力上限量级 ~50 N），也是臂扛不住的冲击。min-jerk ramp + 40 N 饱和两个都要，缺一个都会漏。

**(b) release 条件用位置，不用力阈值。**

```python
release: ‖x_ee − x_hand‖ < 2 cm 持续 0.2 s
```

用力阈值（`‖F‖ < 3N`）会让容差变成 `3/K_h` —— `K_h=100` 时容差 3 cm，`K_h=1000` 时 3 mm。**刚性大的人要求臂让得更精确**，这没有任何道理，纯粹是参数化副作用，而且 `K_h` 是 policy 观测不到的隐变量，等于引入了不可观测的难度波动。

**(c) 没有超时。人是 stubborn 的。**

臂不让开，人就一直推到 episode 结束，success 彻底拿不到。

> 这条最容易被"好心"破坏。任何形式的超时兜底都是 leak：policy 会算账 —— 硬扛只付 force penalty 然后人自己走了，让开要丢 progress 还要花时间回来 —— **然后学会硬扛**，各向异性指标（§7 第 4 条）直接归零，整个实验白做。
>
> Stubborn 的作用是让"让开"成为**严格占优**策略，不依赖 reward 权重去凑。二元决策交给环境结构，不交给权重。

### 4.4 二次推

每个 episode **30% 概率**推第二次，在第一次 release 后 `U(0.5, 1.5)` s 触发，参数重新采样。

目的：逼 policy 学出"刚被推完不要立刻高刚度冲回去"，即对干扰的一点预期。代价几乎为零。

### 4.5 为什么只推末端

推肘部（link 4–5）和推末端（link 7）在 7-DoF 臂上是**完全不同的任务**：

- **推肘部**：臂有 1 维 nullspace，理论上可以肘部让开、末端不动，任务零损失。这时 yield 和 task 不冲突，学的是"把干扰吸收进 nullspace" —— 是 whole-body compliance，更容易。
- **推末端**：没有 nullspace 可用，末端必须真的挪位。这才是真正的 trade-off。

混在一起训，会分不清 policy 学到的是哪个，而且 §7 的评估指标在两种情况下含义不同。

**Stage 1 只推末端**（arbitration 是 downstream 的核心）。推肘部留给 Stage 1.5 —— 那时 joint torque obs 才真正派上用场（推肘部时腕部 F/T 读数为零）。

---

## 5. Obs / Reward

### 5.1 Obs（36 维）

```
q (7), q̇ (7), τ_meas (7), x_ee (3), ẋ_ee (3), (x_g − x_ee) (3), a_{t−1} (6)
```

送入 **GRU，hidden 128**，接 actor/critic head。

**不要喂**：`K_h`、人的状态机状态、`x_hand`、附着 link、`F_ext`、估计后的 `τ̂_ext`。这些必须从 `τ_meas` 历史里在线推断 —— 这个 sysid 结构和 Stage 3 里"从力历史推断 object stiffness / 抓稳没有"是同构的，是 Stage 1 最有价值的迁移点。

> 喂原始 `τ_meas` 而不是 `τ̂_ext`：`τ̂_ext` 依赖动力学模型，模型误差会被算成"外力"，等于把要补偿的误差提前烧进输入里。

**GRU 在 Stage 1 是必需的吗？** 边缘。隐状态只有 `K_h` / 方向 / 是否松手。但 Stage 2/3 一定需要，且序列 buffer、hidden state 管理、部署时序对齐这些工程细节必须先暴露一遍。用 GRU。

Obs 归一化：running mean/std，标准做法。

### 5.2 Reward

| 项 | 形式 | 起始权重 | 单 episode 量级 |
|---|---|---|---|
| progress | `‖x_ee−x_g‖_{t−1} − ‖x_ee−x_g‖_t` | **+10** | ≈ +4 |
| success | 距离 <2 cm 持续 0.5 s → 给奖并结束 | **+1** | +1 |
| time | 常数 | **−0.005** | ≈ −2.5 |
| force | `−max(0, ‖F_ext‖ − 5N)` | **−0.001** | ≈ −3 |
| K | `−mean(log K / log K_max)` | **−0.002** | ≈ −0.5 |
| action rate | `−‖a_t − a_{t−1}‖²` | **−0.01** | 需实测 |

- Reward **可以**用 privileged ground truth（`F_ext` 从 sim 直接取），obs 不行。
- Stage 1 用 `‖F_ext‖`（Newton，可解释，和评估指标同单位）。Stage 1.5 加肘部推时换成 `‖τ_ext‖`。
- **必须逐项 log episodic sum。** 上表的量级是估算，第一次跑完立刻对一遍，任何一项比别的大 100× 就是权重错了，别硬调超参。
- `−λ_K‖K‖` 这项不能删：不罚 K，RL 必然收敛到"到处高刚度"（sim 里高刚度 tracking error 最小、没代价）。这是这类工作最常见的失败模式。

---

## 6. Curriculum

Stubborn 人 + 无超时 ⇒ policy 早期不会让开就永远拿不到 success，reward 极稀疏，可能卡住。用 curriculum 解决，**不要**用加超时解决（超时永久污染 objective，curriculum 只影响训练早期）。

只对 `d` 做 curriculum。`K_h`、方向、`t_push` 从第一步就满量程随机化。

| Level | `d` |
|---|---|
| 0 | `U(2, 5)` cm |
| 1 | `U(2, 10)` cm |
| 2 | `U(3, 15)` cm |
| 3 | `U(5, 20)` cm ← 最终分布 |

规则：

- 全局（不是 per-env），基于最近 100 个 episode 的 success rate。
- `success_rate > 0.7` → 升一级。
- **单调，永不回退。**
- 升到 level 3 后，至少再训 30% 的总步数，保证最终策略在最终分布上收敛。
- **评估永远在 level 3 跑，无论 curriculum 当前在哪一级。**
- Log 当前 level 和升级时刻，画在 reward 曲线上（曲线掉一下是正常的，别以为是崩了）。

---

## 7. 评估协议

固定 200 个 seed（goal / `K_h` / `u` / `d` / `t_push` 全部固定），所有对比对象跑同一批。**Level 3 分布。**

### 主指标

1. **Yield ratio** = 推的过程中臂沿 `u` 方向的位移 / `d`。
   硬扛 ≈ 0，完全顺从 ≈ 1。arbitration 的直接度量。

2. **Peak `‖F_ext‖`**（N）。对比解析 baseline。

3. **Recovery time**（s）= release 到重新 success 的时间。

4. **K anisotropy ratio** = `K_∥ / K_⊥`，在 pushing 窗口内取均值。
   ```
   K_∥ = Σᵢ uᵢ² Kᵢ            # 沿推力方向的等效刚度
   K_⊥ = (trace(K) − K_∥) / 2  # 正交方向平均
   ```

5. Success rate、episode 时长。

### ⚠️ 第 4 条是成败的真正判据

> 如果 policy 只是把三个轴一起调软（`K_∥/K_⊥ ≈ 1`），那它没学会 arbitration，只学会了"感到力就整体变软" —— **这个用解析律就能做，白训了。**
>
> 要的是 `K_∥/K_⊥` 明显 < 1：沿干扰方向软、正交方向保持刚度继续朝 goal 推进。"顺从"和"完成任务"不是二选一，而是在不同方向上同时进行。这正是解析 admittance 做不到的事。

期望的三段行为：
1. **推的过程中**：沿 `u` 软、顺着走；正交方向保持刚度，继续朝 goal 推进 ← 全部重点
2. **松手瞬间**：不立刻高刚度弹回（force penalty + action rate 负责压）
3. **松手之后**：重新收敛到 goal

---

## 8. 实验清单

### E0 — 环境 sanity check（先做，不训任何东西）

- [ ] 无干扰下，解析 impedance 能到 goal，无超调震荡
- [ ] 人推的时候 `F_ext` 剖面合理：ramp 上升、峰值 ≤ 40 N、无冲击尖峰
- [ ] `K_h = 1000, d = 20cm` 极端情形下臂不发散、不报错
- [ ] Stubborn 验证：把 policy 换成"输出恒定高 K"，确认 **episode 永远不成功**。这条不过就是 release 条件写漏了。
- [ ] Fallback 路径：注入 NaN / 超时 / 越界，确认退回纯解析 impedance。**在写 policy 之前先测通。**
- [ ] Log 一条 episode 的所有 reward 分项，对照 §5.2 的量级估算

### E1 — 解析 baseline

固定 K 的 Cartesian impedance（无 policy），扫 3 个刚度：`K ∈ {200, 600, 1500}` N/m 各向同性。
跑完整评估协议。**这是所有 policy 必须超过的对象**，特别是要证明 policy 拿到了单一各向同性 K 拿不到的 (yield ratio, recovery time, peak force) 组合。

### E2 — 主训练

PPO + GRU + curriculum，3 个 seed。
起始超参：`lr 3e-4`, `γ 0.99`, `λ 0.95`, `clip 0.2`, `entropy 0.005`, 64 并行 env, BPTT 截断长度 32。
总步数：先跑 3e7，看曲线决定要不要加。

### E3 — time / force 权重扫描（这是唯一真正要调的东西）

3×3 网格，各 1 seed：

```
w_time  ∈ {−0.002, −0.005, −0.0125}
w_force ∈ {−0.0004, −0.001, −0.0025}
```

产出**行为相图**：横轴 `w_force/w_time`，纵轴 yield ratio + K anisotropy。预期一端硬扛、另一端躺平不完成任务，中间是目标带。

> 有了 stubborn 人之后，这两项不再决定"让不让"（环境结构已经决定了），只决定"让得多激进/多快"。这个调参直觉能直接搬到 Stage 2/3。

### E4 — Ablation

| ID | 改动 | 检验什么 |
|---|---|---|
| A1 | K 输出退化成标量（各向同性） | 各向异性是不是收益来源。**最重要的一个** |
| A2 | GRU → MLP（单帧） | history 是否必需（预期：`K_h` 推不出来，性能下降但不崩） |
| A3 | 无 curriculum，直接 level 3 | curriculum 是否必需 |

### E5 — 负对照（诊断用，1 seed）

给人加回 **2 s 超时**（即故意注入 §4.3(c) 说的 leak）。

**预期：policy 学会硬扛，yield ratio → 0，anisotropy → 1。**

如果这个预期没复现，说明"超时是 leak"的推理有问题，§4.3(c) 的设计依据要重新审视。这条便宜、快、且能证伪自己的设计假设，值得跑。

---

## 9. Domain randomization

只加这些（都是 Stage 2/3 同样存在、随机化范围有复用价值的）：

- control latency `U(0, 10)` ms
- joint torque sensor：高斯噪声 + per-episode 常数 bias
- payload `U(0, 2)` kg（末端）

**不加**别的。特别是不要加接触相关的随机化 —— Stage 1 没有接触。

---

## 10. 代码结构

```
stage1/
  envs/
    rizon_base.py        # MuJoCo 加载、1kHz impedance 层、fallback  ← Stage 2 复用
    human.py             # §4 状态机                                  ← Stage 2 改造复用
    obs.py               # §5.1 obs 构造                              ← Stage 2 复用
    action.py            # §3.2 decode_action，独立可替换            ← Stage 2 会改
    reward.py            # §5.2                                       ← Stage 2 重写
    task_reach.py        # goal 采样、success 判定                    ← Stage 2 重写
    curriculum.py        # §6
  train.py               # PPO + GRU                                  ← Stage 2 复用
  eval.py                # §7 全部指标                                ← Stage 2 大部分复用
  baselines/analytic.py  # E1
  deploy/                # fallback + 插值 + 时序对齐（现在就写，Stage 2 用）
  slurm/
```

标 "Stage 2 复用" 的文件里**不允许出现 reach 任务或人模型的假设**。这是 §0 判据的执行方式。

---

## 11. 计算 / SLURM

集群 `bos14`，两个 partition（注意命名坑）：

| partition | 实际 GPU | gres |
|---|---|---|
| `gpu_a100` | **RTX PRO 6000**（不是 A100） | `gpu:rtxpro6000:1` |
| `gpu_5090` | RTX 5090 | `gpu:rtx5090:1` |

- **一律 `sbatch`，不要用交互式 `srun`/`salloc`。**
- MuJoCo 需要 `export MUJOCO_GL=egl`。
- E3 的 3×3 扫描和 E4 用 job array + 节流：`sbatch --array=0-8%9 ...`。
- 训练本身是 CPU-heavy（MuJoCo 物理）+ 小网络，先用 CPU 向量化 env（`--cpus-per-task=16`），别一上来就上 MJX —— MJX 重写 1 kHz impedance 层和 GRU PPO 的成本远超收益，除非吞吐真的成为瓶颈。

---

## 12. 失败模式排查表

| 现象 | 大概率原因 | 处理 |
|---|---|---|
| `K_∥/K_⊥ ≈ 1` | 没学会 arbitration | 查 release 是不是漏了超时；`w_force` 加大；确认 `u` 采样真的各向均匀 |
| yield ratio ≈ 0 | 硬扛 | 一定是 stubborn 破了。跑 E0 的 stubborn 验证 |
| 所有轴 K 顶到 2000 | `λ_K` 太小 | 加大；确认 K penalty 真的接进了 reward |
| 早期完全不成功 | curriculum 没生效 | 查 level 是否卡在 0；`d` 下限降到 1 cm |
| 训练崩 / K 剧烈抖动 | action rate 太小，或 D 没绑定 | 加大 `w_action_rate`；确认 `D = 2ζ√(Km)` |
| 成功但 recovery 极慢 | `w_time` 太小 | E3 扫描里找 |
| 松手瞬间弹飞 | action rate / force penalty 不够 | 两个都加大 |

---

## 13. 交付物

1. 上述代码
2. E1–E5 全部结果，统一评估表（§7 五个指标 × 所有对比对象）
3. E3 的行为相图
4. 一段 rollout 视频：能看到"沿推力方向让开、同时正交方向朝 goal 走"
5. **一份 Stage 2 复用性报告**：§10 里标"复用"的文件，实际有多少行需要为 Stage 2 改动。这是 §0 判据的答案。

---

## 14. 需要确认的开放问题

- **[阻塞 Stage 2，不阻塞 Stage 1]** Flexiv RDK 能不能拿到**逐关节的原始 torque 读数** `τ_meas`，而不只是估计后的末端 wrench 或 `τ̂_ext`？整个 obs 设计建立在这个前提上。Stage 1 在 sim 里无所谓，但如果 SDK 只给估计量，Stage 2 上真机时 obs 要重新设计 —— **在开始写 Stage 2 之前先查 SDK 文档确认掉。**
- Rizon 4 的 MuJoCo 模型来源和惯性参数可信度（影响 Stage 2 迁移，不影响 Stage 1 结论）。
