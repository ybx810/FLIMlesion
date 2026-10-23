# FLIM RGB 批量降采样

独立入口：`D:\FLIMlesion\scripts\downsample_images.py`。只读取原图，不改动现有数据或其他项目代码。默认扫描所有子目录，扩展名大小写不敏感，默认输出 3 种方法 × 1024/2048/4096 共九组无损 PNG。

## 环境与 PowerShell 命令

本机已建立独立环境 `D:\FLIMlesion\.venv-downsample`，避免影响已有 Python 环境。系统 `python` 指向不可用的 Windows 应用别名，因此以下命令直接使用解释器路径，无需激活环境。

首次安装/在另一台机器重建（Python 3.11）：

```powershell
Set-Location D:\FLIMlesion
& D:\miniforge3\envs\py311\python.exe -m venv .venv-downsample
& .\.venv-downsample\Scripts\python.exe -m pip install -r .\scripts\requirements-downsample.txt
```

合成图像回归测试（临时目录，不使用真实数据）：

```powershell
& D:\FLIMlesion\.venv-downsample\Scripts\python.exe -m unittest discover -s D:\FLIMlesion\scripts -p test_downsample_images.py -v
```

只预览前两张图的任务；读取头部，不解码像素、不创建输出目录或日志：

```powershell
& D:\FLIMlesion\.venv-downsample\Scripts\python.exe D:\FLIMlesion\scripts\downsample_images.py --limit 2 --dry-run
```

小规模试运行，包括一张原图的三算法对照图：

```powershell
& D:\FLIMlesion\.venv-downsample\Scripts\python.exe D:\FLIMlesion\scripts\downsample_images.py --limit 2 --comparisons 1 --comparison-size 1024
```

正式运行全部数据、全部九组：

```powershell
& D:\FLIMlesion\.venv-downsample\Scripts\python.exe D:\FLIMlesion\scripts\downsample_images.py --input-dir D:\FLIMlesion\data --output-dir D:\FLIMlesion\outputs\downsampled --methods all --sizes 1024 2048 4096
```

选择算法/尺寸：`--methods area gaussian_pyramid --sizes 2048 4096`。默认跳过已存在输出；明确重新生成时追加 `--overwrite`。不要同时启动多个批处理进程。`--threads 1` 为 OpenCV 内部线程数，始终逐张处理；不是多图并行参数。`--limit` 在稳定排序后截取。`--memory-budget-gb 18` 为默认估算峰值预算，单位 GiB。

## 算法与尺寸

三种方法都直接使用同一张原图解码得到的只读 RGB 数组，不互相引用输出。无额外锐化、颜色增强、均衡化、白平衡、伽马或去噪。

| 方法 | 实际实现 |
|---|---|
| lanczos | `cv2.resize(..., interpolation=cv2.INTER_LANCZOS4)`，标准 8×8 邻域 Lanczos4，保留其原生插值和 uint8 舍入行为 |
| area | `cv2.resize(..., interpolation=cv2.INTER_AREA)`，真实像素区域平均 |
| gaussian_pyramid | 逐级 `cv2.pyrDown`，5×5 核为 `[1,4,6,4,1]` 的外积除以 256，边界 `BORDER_REFLECT_101`；每级宽高为前级向上取整的一半；下一层长边小于目标时停止，然后必要时 `INTER_LINEAR` 精确调整，绝不先缩过小再放大 |

参考：[OpenCV resize](https://opencv.org/resizing-and-rescaling-images-with-opencv/)、[OpenCV pyrDown](https://docs.opencv.org/4.11.0/d4/d86/group__imgproc__filter.html)。

尺寸以完成 EXIF Orientation 校正后的宽高为准；长边严格等于目标值，短边统一采用整数 round-half-up（四舍五入，最少 1 像素）。整数栅格造成的短边误差不超过 0.5 像素，极端细长图最小 1 像素是例外。不裁剪、不补边、不拉伸为正方形。小于目标则跳过，等于目标允许输出；高斯无需调整时日志记 `none`。例如 30000×20000 输出 1024×683、2048×1365、4096×2731。

金字塔每一级为 uint8；量化舍入属于 OpenCV 原生行为。Lanczos4 的大倍率缩小行为与 Area 的抗混叠行为不同，这是本实验比较的算法差异，程序没有偷偷增加滤波。

## 输入颜色策略

- 统一 Pillow 解码，统一 `ImageOps.exif_transpose`，输出不带过期 Orientation。
- 只接受单帧 8-bit RGB。检查模式、通道数、PNG 原始 IHDR 位深和 TIFF BitsPerSample；尤其防止 Pillow 将 RGB16 PNG/TIFF 暴露为 RGB8 后静默降位深。
- 16-bit、灰度、调色板、CMYK、Alpha、PNG 透明色、多页 TIFF/动画图像明确失败，记录原因，不隐式转 RGB、合成背景或丢弃通道。需要这些格式时应先制定独立的科学转换规范。
- ICC 原样复制到输出 PNG，日志记录 SHA-256；不做 ICC 色彩空间转换。无 ICC 不假设或新增配置。不同来源的 ICC/编码空间可能不同，跨来源分析应检查日志；插值在原有编码 RGB 数值上完成，不进行线性光变换。
- 不通过 OpenCV 读取或保存图像，不涉及 BGR 转换；其滤波对通道独立，所以数组一直保持 RGB 顺序。
- 像素间距、原始设备参数等未被推断或重建，PNG 不携带可用于 Radiomics 的新物理标定；物理尺度与 ROI 需在后续分析中明确设置。叠加 RGB 不能被解释为三个独立原始物理通道。

## 超大图像与内存

本版本使用完整解码，不声称流式处理；Pillow + OpenCV 保证算法定义明确，不使用 pyvips 近似替代。每张只解码一次，顺序生成各输出，释放输出数组和高斯中间层，结束后关闭图像并回收对象；不缓存所有层或所有输出。只有对照图需要额外画布。

解码前估算峰值：`16 × 原图像素数 + 32 × 最大请求目标长边² + 256 MiB`。覆盖解码、方向校正、NumPy 复制、处理和 PNG 保存的常见内存占用，默认阈值 18 GiB。30000×30000、最大目标 4096 的估算约 14.16 GiB，可进入处理；40000×40000 估算约 24.59 GiB，默认拒绝并记录 failed。预算是保守规划值，不是操作系统硬限制，不保证任意压缩格式和库版本的真实峰值；应给 Windows 和其他软件留空间。超预算不会偷偷改变算法或先用 JPEG 缩略解码。

Pillow 默认像素数保护仅在头部打开阶段临时解除，替代为显式内存预算。Python 可捕获的内存不足、解码异常、写入异常逐任务记录并继续；操作系统直接终止进程无法由 Python 捕获。极大图若超出本机内存预算，应在更大内存设备运行，或另行实现经数值验证的分块算法。本版本未进行数万×数万真实图像峰值压力测试。

## 输出、命名与审计

- 数据：`D:\FLIMlesion\outputs\downsampled\{method}\{size}\{原始相对目录}\{stem}__{扩展名}__{SHA256}.png`。
- SHA-256 来自大小写保留的相对路径，保留原扩展名，主文件名最长取 72 字符；新加文件或改变 `--limit` 不改变已有映射。支持中文、空格路径。极深目录仍受 Windows 长路径配置影响，写入错误会记录。
- 输入与输出树必须分离，拒绝输出位于 data 内或反向包含 data。先写同目录临时 PNG，完成后发布；默认独占创建，`--overwrite` 才替换。默认跳过只按存在性判断，不验证旧输出正确性；改参数或库版本后建议新输出目录。
- 日志：`processing_log.csv`，UTF-8 BOM，追加写入，每个任务一行且立即刷新。包含用户要求的所有字段，以及输入/输出模式、位深、ICC 哈希、EXIF、估算内存、库版本、算法参数和高斯实际层级尺寸（宽、高）。读取失败时无法获取的字段留空。
- `processing_time_seconds` 为单输出任务计算/写入时间；共享解码耗时单列 `decode_time_seconds`（同源各行重复，不应逐行相加）。汇总总时间包含扫描、解码和对照图。
- `run_*.json` 保存每次运行参数、Python/NumPy/Pillow/OpenCV 版本与尺寸规则。正式运行若有任何失败返回退出码 1，参数错误为 2，全部成功或跳过为 0。
- 对照图：`comparisons\{size}\{原始相对目录}\*.png`；按 Lanczos、Area、Gaussian Pyramid 横向排列，面板使用精确目标像素，不再次缩放，标签带方法/尺寸及 VISUAL ONLY。顶部标签区不属于实验图。对照图不是九组数据之一，禁止用于分类或定量分析；下游仅读取指定方法/尺寸文件夹。对照图失败记录在 `comparison_errors.jsonl`，不改变九组任务计数。
- `--dry-run` 将任务 JSON 和汇总打印到控制台，状态为 planned/跳过/failed，不生成日志或图像，不证明完整解码一定成功。

## 测试范围

`test_downsample_images.py` 使用 unittest，覆盖正方形/非正方形/单像素短边、三档目标尺寸计算、与 OpenCV 参考算法逐像素一致、只读源数组不变、RGB 颜色顺序、文件哈希不变、中文空格路径、同主文件名跨扩展名、跳过/覆盖、坏文件后继续、ICC/EXIF、16-bit/Alpha 拒绝、内存拒绝、dry-run 不落盘及对照图尺寸。

真实数据试运行的原图 SHA-256 保存在 `D:\FLIMlesion\outputs\downsample_source_hashes.json`。仅试运行少量图像；完整数据集请使用上述正式命令。
