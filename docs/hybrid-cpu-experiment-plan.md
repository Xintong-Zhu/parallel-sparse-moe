# 下一组实验：固定 256 核的 MPI + OpenMP 配比

状态：运行脚本已实现，尚未产生本组性能结果。根据 1539728 的 EP/DP 数据制定。

提交方式（Vanda 登录节点）：

```bash
bash scripts/submit_vanda_cpu_formal.sh scripts/vanda_cpu_hybrid.pbs
```

日志位于 private/hybrid-cpu-<jobid>/job.log。只有 job_status.txt 中 exit_status=0、
stage=complete 且 20 个测量文件齐全，才视为完成；提交脚本使用 Git 冻结源码。

## 问题与假设

EP 在 128×1 时为 0.0339162 秒，在 256×1 时为 0.0687345 秒。
下一组检验：固定 256 个实际计算核，通过减少 MPI 进程并增加每进程线程，
能否降低 EP 的通信成本，同时保留足够的 Expert 计算并行度？
加入 DP 作为对照，帮助观察 OpenMP、批量任务与内存访问开销。
DP 与 EP 的计算组织不同，不能把两者之差全部归因于网络。

## 实验矩阵

| MPI 进程 P | 每进程线程 T | P×T | 每节点进程 | 每插槽进程 | 每进程 EP Experts |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 256 | 64 | 32 | 1 |
| 128 | 2 | 256 | 32 | 16 | 2 |
| 64 | 4 | 256 | 16 | 8 | 4 |
| 32 | 8 | 256 | 8 | 4 | 8 |
| 16 | 16 | 256 | 4 | 2 | 16 |

每种配比各测 EP 与 DP，共 10 个配置。EP 固定 Alltoallv、轮询 Expert 放置。
DP 复制全模型权重，不在计时前向内执行 Token all-to-all 或全局输出收集。

所有点冻结 N=131072、E=256、d=128、h=256、Top-2、uniform trace。
种子保持 model=42、input=314159、routing=271828、calibration=161803。
OpenMP 使用 static；OMP_DYNAMIC=FALSE、OMP_PLACES=cores、OMP_PROC_BIND=close。
MPI 调用继续只由主线程执行，保持 MPI_THREAD_FUNNELED。
模型、输入、路由和 oracle 都在计时外准备，沿用 moe_cpu_bench 的前向计时边界。

## 资源与绑定

申请四个同型号、双插槽、每插槽 36 个物理核的节点。建议完整申请每节点 72 核，
保留每节点 64 个 MPI 启动槽；每节点实际计算只用 64 核，每插槽用 32 核。
因此 PBS 申请 288 核，算法使用 256 核。报告同时列出申请与使用数量；
计费按集群规则和实际申请资源计算，不能写成仅申请 256 核。

原因：原来的 64 核申请可能允许 36+28 核的插槽分配；每进程 16 核的绑定
在这种分配下不能按每插槽两个进程对称安排。原始日志中的单核 slot 映射也
不是新设计的对称插槽映射，所以旧作业结果仅作背景，新实验必须重测基线。

候选 PBS 资源请求（需按实际分配核验）：

```bash
#PBS -l select=4:ncpus=72:mpiprocs=64:ompthreads=1:mem=250gb
#PBS -l place=scatter:excl
#PBS -l walltime=01:00:00
```

运行时逐配置显式设置 OMP_NUM_THREADS=T；PBS 中的 ompthreads=1 是初始值。
MPI 映射按每插槽 K=32/T 个进程，每进程绑定 T 个核：

```bash
mpirun --map-by ppr:K:package:PE=T --rank-by slot --bind-to core -np P ...
```

K、T、P 是占位符，提交脚本需要替换为矩阵中的整数。
Open MPI 的 package/PPR/PE 映射语义参见
[官方文档](https://docs.open-mpi.org/en/v5.0.6/man-openmpi/man1/mpirun.1.html)。
具体运行环境仍为 Vanda gompi/2024a（Open MPI 5.0.3），必须先通过该环境下的探针。

正式计时前的硬性条件：四节点 CPU 型号与拓扑相同；五种配比都启动正确数量
的 MPI ranks 和 OpenMP threads；每个进程实际绑定 T 个物理核；同一节点上
进程的核集合不重叠；五种配比的计算核集合相同；每节点合计 64 核、总计 256 核。
保留 PBS 分配、lscpu、rank/thread CPU affinity 和 mpirun binding 输出。
launch_probe 的 --require-hybrid-layout 会读取每个线程实际 affinity mask 与 Linux
sysfs 的 package/core ID，检查物理核不重叠、进程不跨插槽、每节点 64 核、
每插槽 32 核，并输出物理核集合；脚本比较五个集合完全一致才计时。
若绑定不满足条件，停止并修正资源分配，不使用 oversubscribe。

## 执行顺序与重复

在同一 PBS 作业、同一冻结源码中完成所有配置。
先完成四节点通信及五种配比的启动和亲和性探针，再运行性能矩阵。

- Round A：256×1、128×2、64×4、32×8、16×16；每个配比先 EP 后 DP。
- Round B：16×16、32×8、64×4、128×2、256×1；每个配比先 DP 后 EP。
- 每次程序调用预热 3 次、测量 5 次，共 20 次调用、100 次计时。
- 每次都与相同输入的串行 oracle 校验；每条计时必须 bitwise_ok=1。
- 两轮分别报告中位数和 min/max，也可汇总每配置的全部 10 次计时，不能只挑较快一轮。
- 如果两轮排序不一致或中位数相差超过 10%，标记不稳定，并在时间允许时补测
  两个有争议的配置及其基线；不把额外轮次当成独立机器重复实验。

每个点保存独立原始文件，并记录 round、P、T、Git commit、PBS jobid、节点、
配置、种子、Expert/进程任务数及正确性。日志目录建议 private/hybrid-cpu-<jobid>/。
EP 均匀负载应为每 Expert 1024 个 assignment、每进程 262144/P 个 assignment。
线程任务数依实际调度核验，不能仅用这个平均值声称所有线程负载完全一致。

## 指标与解释

主指标是各进程前向耗时的最大值。分别记录 Expert 计算、派发与返回通信、
打包、解包与合并时间。阶段最大值可能来自不同 rank，不相加为严格总时间占比。

固定核数的相对加速定义为 S(P,T)=T(256,1)/T(P,T)，EP 和 DP 各自使用
本轮同模式基线。该指标衡量混合配置收益，不再除以 256 称为扩展效率。
正式强扩展效率继续使用此前明确的一核基线。

图表：

1. 横轴五种 P×T 配比，纵轴时间（秒，对数刻度），EP/DP 两条曲线或分组点，显示重复范围。
2. 横轴同配比，纵轴相对各自 256×1 的收益，注明基线。
3. EP 的计算、派发通信、返回通信分别画线，不把 rank 最大值堆叠成总耗时。

假设而非预定结论：减少 MPI 参与者可能改善 EP 通信，但每进程消息更大，
本地打包/合并串行工作更多，也会增加 OpenMP 调度与同步开销。DP 若也变慢，
可提示本地线程或内存开销；EP 若通信下降、整体仍不改善，则要看这些本地阶段。
不能仅凭这组实验断言网络拥塞、缓存命中率或 NUMA 是已证实根因，后者需要额外测量。

本组之后再做 16 Expert、固定资源下的 Alltoallv/P2P 与均匀/偏斜负载比较。
负载感知放置需先实现并验证，不能把未实现的选项计入已完成实验。
