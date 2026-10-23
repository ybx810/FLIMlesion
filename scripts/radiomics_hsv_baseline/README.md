# Gray / RGB / Gray + HSV 八分类 Baseline

## 本次增量扩展

三种方案统一使用已有 RF / XGBoost、随机种子 42、患者级固定交叉验证和折内预处理。`all` 仅包括以下三组，没有 HSV Only 分类入口。

| feature-set | 输入特征 | 原始特征数量 |
|---|---|---:|
| gray_radiomics | 灰度 Original Radiomics | 93 |
| rgb_radiomics | R/G/B 各自 Original Radiomics | 279 |
| gray_radiomics_hsv | 灰度 Radiomics + 原 HSV/颜色区域特征 | 169 |

只启用 firstorder、glcm、glrlm、glszm、gldm、ngtdm；无 shape、Wavelet、LoG。RGB 是设备导出伪彩色融合图的显示颜色通道，不代表三个独立原始光学信号。

## 当前数据与五折划分

当前九组共有 **187 张图、115 位患者、8 类**。现在仅保留既有患者级 **5-fold** 划分：`outputs/radiomics_hsv_baseline_5fold/cv_splits/cv_splits.csv`，fold 编号为 0–4。已核对当前队列完整覆盖且无患者泄漏，没有重新生成划分。

`config.yaml` 和 `config_5fold.yaml` 均使用五折，输出到 `outputs/radiomics_hsv_baseline_5fold`。三折配置 `config_3fold.yaml` 及旧输出目录下的三折划分文件已删除。旧三折实验结果已按要求删除。

新入口必须有已有划分：若更换输出目录，应通过 `cv.existing_splits` 指向上述五折文件，缺失则报错，绝不自动生成新划分。

输入目录优先保持实际名称 `gaussain_pyramid`，也兼容已存在的 `gaussian_pyramid`。本机当前实际目录为后者，运行时记录警告，未重命名。全部9组通过 metadata 和已有 processing_log 精确匹配；不同分辨率保持相同 sample_id。

## PowerShell 运行

使用已安装的 Python 3.9 环境；CPU 计算，无 CUDA。工作目录为 `D:\FLIMlesion`。

```powershell
Set-Location D:\FLIMlesion
# 让本地 conda 环境的 DLL 可用；限制计算库线程，避免嵌套并行
$env:PATH = "$PWD\.conda-radiomics;$PWD\.conda-radiomics\Library\bin;" + $env:PATH
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

# area/1024，三种特征 × 两种模型
.\.conda-radiomics\python.exe -m scripts.radiomics_hsv_baseline.run --config scripts/radiomics_hsv_baseline/config_5fold.yaml --stage all --method area --resolution 1024 --feature-set all --n-jobs 4

# 单组 RGB：仅提取 / 仅分类
.\.conda-radiomics\python.exe -m scripts.radiomics_hsv_baseline.run --config scripts/radiomics_hsv_baseline/config_5fold.yaml --stage extract --method area --resolution 1024 --feature-set rgb_radiomics
.\.conda-radiomics\python.exe -m scripts.radiomics_hsv_baseline.run --config scripts/radiomics_hsv_baseline/config_5fold.yaml --stage train --method area --resolution 1024 --feature-set rgb_radiomics --classifier rf

# 全54组：9种图像配置 × 3种特征 × 2种分类器
.\.conda-radiomics\python.exe -m scripts.radiomics_hsv_baseline.run --config scripts/radiomics_hsv_baseline/config_5fold.yaml --stage all --feature-set all --classifier all --n-jobs 4

# 九组数据发现，不提取、不训练、不写结果
.\.conda-radiomics\python.exe -m scripts.radiomics_hsv_baseline.run --config scripts/radiomics_hsv_baseline/config_5fold.yaml --dry-run --feature-set all

# 回归测试
.\.conda-radiomics\python.exe -m unittest scripts.radiomics_hsv_baseline.test_pipeline scripts.radiomics_hsv_baseline.test_baselines -v
```

参数：`--stage qc/extract/train/all`；`--method area/lanczos/gaussain_pyramid/gaussian_pyramid`；`--resolution 1024/2048/4096`；`--classifier rf/xgboost/all`。`--feature-set` 默认 `gray_radiomics_hsv`，兼容旧别名 `fused`、`radiomics`，不再接受 `hsv`。`--limit N` 仅用于 qc/extract，独立写入 `_smoke_N`，不得将 smoke 表用于训练。

为了保护已完成的结果，新入口不允许 `--force`。当输入、ROI、参数、版本或已验证文件发生变化时明确报错；需要新研究方案时指定新的 output_root 和既有 cv.existing_splits。不要同时向同一输出目录启动多次命令。

## 提取、复用与内存

保留原 `radiomics.csv`、`hsv.csv`、`fused.csv` 和旧逐样本缓存，新增三种具名特征表。旧缓存复用需验证原提取实现兼容性、依赖版本、配置、每张图/掩膜 SHA256、sample_id/patient_id/label、表文件哈希，以及 radiomics 表和 fused 表的 RAD_ 列数值完全一致。显式投影到当前共有队列的过程写入 `legacy_reuse_audit.json`。

RGB 使用独立 `rgb_cache/`，逐张读取、三个通道依次提取，三个通道复用同一个 ROI；每个通道计算后释放中间数组。单通道共享原灰度提取函数，设置为 binWidth=16、distance=[1]、force2D=True、force2Ddimension=0、Original；不缩放、不重采样、不归一化。单层 SimpleITK 容器外补一圈零背景，保留全部原图像素。未校准物理尺度，spacing=1 pixel。

OpenCV、SimpleITK 单线程，模型默认4线程；不并行加载整组图像。日志逐样本显示进度。常量通道警告、通道提取失败、NaN/Inf 写日志；提取失败汇总到 `rgb_extraction_errors.csv` 并阻断训练，不能静默删样本。成功逐样本缓存可用于中断后续跑。HSV 的正常缺失（例如无颜色区域）保持原定义，在训练折内处理。

ROI 优先使用配置的同尺寸二值掩膜，默认全图；原 HSV 阈值、黄色/粉色/绿色区域规则均未改动。图像读取继续复用原降采样模块的 RGB8/EXIF/ICC 检查。

## 模型和输出

复用 `train_classifiers.py`、`evaluation.py`，未改变模型超参数。训练折独立拟合全缺失列剔除、中位数填补、方差过滤，以及已有的可选相关性过滤/标准化；默认不启用后两项。sample_id、patient_id、label 永不进入模型。八类按原 sorted(label) 编码。

每个 sample_id 只获得一次 OOF 预测；检查概率有限、和为1、全部样本覆盖，以及患者不跨 fold。最终指标来自图像级 OOF，患者是划分单位。既有完整实验只有在特征数值、队列、fold、参数和版本完全一致时才复用；否则独立存储新实验，绝不覆盖旧融合目录。

```text
outputs/radiomics_hsv_baseline_5fold/
  features/{实际方法}_{分辨率}/
    radiomics.csv, hsv.csv, fused.csv       # 保留旧缓存
    gray_radiomics.csv + .json
    rgb_radiomics.csv + .json
    gray_radiomics_hsv.csv + .json
    rgb_cache/*.json
  cv_splits/
    cv_splits.csv                         # 保留的五折划分
    comparison_cv_splits.csv               # 五折实验运行时保存的已验证副本
    comparison_split_provenance.json
  experiments/{配置}/
    rf/, xgboost/                         # 历史融合结果保留
    {feature_set}/{rf或xgboost}/           # 新三组实验
  comparisons/
    baseline_comparison.csv
    baseline_per_class.csv
    comparison_contract.json
    *.png, *.pdf
```

每个实验输出 metrics.json、classification_report.csv、oof_predictions.csv、confusion_matrix.csv/PNG/PDF、confusion_matrix_normalized.csv/PNG/PDF、feature_importance.csv/PNG/PDF、每折模型、类别映射、折内实际保留特征列表及哈希溯源。

比较表包含全部要求指标；对比图包括三组 Macro-F1、六张方法×分辨率热图、RF/XGBoost 对比及八类 F1 对比。全部使用 0–1 标度和固定类别顺序；尚未运行的格子标记 not run，不填造结果。同一患者划分的分批运行按配置键合并，不覆盖先前已完成配置的比较行。

特征数量 93/279/169 为过滤前输入数量，训练折内可能剔除常量或全缺失特征，实际维数保存在 folds.json。重要性仅为训练折内置重要性均值，不能直接解释为稳定生物标志物。

## 当前结果

area/1024 的六组五折结果见 outputs/radiomics_hsv_baseline_5fold/comparisons/baseline_comparison.csv。旧三折结果及指标表已删除。
