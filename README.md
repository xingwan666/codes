# Tire Pressure OCR Web App

一个基于 `Flask + EasyOCR + OpenCV` 的胎压识别小工具。

它的目标很直接：
- 上传一张飞机轮胎压力页面照片
- 自动识别 14 个胎压数值
- 自动填充前端输入框
- 再根据业务规则输出简单的检查建议

当前项目更偏向“内部工具 / 原型验证”，已经能跑通完整流程，但识别效果仍然依赖图片质量、拍摄角度和设备性能。

## 功能概览

- 图片上传后自动做 OCR 识别
- 识别结果自动回填到 14 个胎压输入框
- 支持前轮和主轮的规则校验
- 针对数字类界面做了多步图像预处理
- 优先做显示区域检测，再做小块数字识别，减少整图重复 OCR
- 返回调试信息，方便继续调参

## 项目结构

```text
.
├─ app.py
├─ templates/
│  └─ index.html
├─ tire press/
│  └─ index.html
├─ uploads/
└─ README.md
```

说明：
- `app.py`：后端主程序，包含 OCR、后处理、规则校验和 Flask 路由
- `templates/index.html`：Flask 默认使用的前端页面
- `tire press/index.html`：一份额外的静态页面副本
- `uploads/`：当前放了一些样图，主要用于本地测试

## 运行环境

推荐环境：
- Python `3.13`
- Windows 本地调试，或 Ubuntu 20.04 服务器


不建议直接用非常新的 Python 版本盲跑，比如 `3.14`，因为部分 OCR 相关依赖兼容性可能不稳定。

## 依赖

建议安装这些包：

```txt
Flask
easyocr
opencv-python-headless
numpy
Pillow
torch
torchvision
gunicorn
```

如果是在 Windows 本地调试，也可以用：

```txt
opencv-python
```

替代：

```txt
opencv-python-headless
```

## 本地启动

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux:

```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install --upgrade pip
pip install Flask easyocr opencv-python numpy Pillow torch torchvision
```

如果是在 Ubuntu 服务器上，建议把 `opencv-python` 换成：

```bash
pip install Flask easyocr opencv-python-headless numpy Pillow torch torchvision gunicorn
```

### 3. 启动服务

```bash
python app.py
```

默认监听：

```txt
http://127.0.0.1:5000
```

## 页面使用方式

1. 打开首页
2. 上传胎压页面截图
3. 等待自动识别
4. 检查自动填入的 14 个胎压值
5. 必要时手工修正
6. 提交规则校验，查看建议结果

## 后端接口

### `GET /`

返回前端页面。

### `POST /ocr_fill`

上传图片并执行 OCR 识别。

表单字段：
- `image`：图片文件

返回示例：

```json
{
  "success": true,
  "tire_pressures": [213, 212, 213, 211],
  "recognized_count": 4,
  "debug": {
    "raw_recognitions": [],
    "total_raw": 0,
    "merged_unique": 0,
    "valid_tire_data": 0,
    "strategies_used": []
  }
}
```

### `POST /check_rules`

提交 14 个胎压值，执行业务规则校验。

请求示例：

```json
{
  "pressures": [213, 212, 213, 211, 213, 214, 214, 212, 215, 218, 212, 211, 213, 214]
}
```

## 当前识别策略

`app.py` 里的识别流程大致是：

1. 读取上传图片并自动处理 EXIF 方向
2. 对图片缩放到适合 OCR 的尺寸
3. 先检测可能的显示区域
4. 对候选区域生成少量高价值预处理版本
5. 先做文本框检测
6. 对每个小框做数字识别
7. 合并、去重、按胎压范围过滤
8. 输出最多 14 个结果

相比最初“整图多轮 OCR”的方式，现在已经减少了很多重复计算，更适合 CPU 环境。

## 已知问题

这个项目现在可用，但还不能算彻底稳定，主要有这些限制：

- 识别率仍然受拍摄角度、反光、模糊影响
- 某些图片虽然能凑齐 14 个值，但顺序或个别数字可能有偏差
- 目前仍然依赖 `EasyOCR + torch + CPU`，单次识别耗时不算低
- 1 核 1G 的小服务器可以跑，但只适合低并发

## 本地实测情况

这版代码在当前样图上做过实测，已经比最初版本更快，也更容易补齐 14 个数值。

其中几张图的结果大致在：
- `8s ~ 13s` 完成一张图的识别
- 多张样图可以识别到 `14/14`

但要注意：
- 这里说的是“数量能补齐”
- 不是“每张图都绝对零误差”


### gunicorn 示例

```bash
gunicorn -w 1 -b 127.0.0.1:5000 --timeout 120 app:app
```

## 后续建议

如果要继续把识别效果往上推，优先级建议是：

1. 增加 `requirements.txt`
2. 增加 `DEPLOY.md`
3. 增加离线测试脚本，批量跑样图统计识别结果
4. 针对固定胎压页面布局做专用槽位识别，而不是只依赖通用 OCR
5. 统一前端页面，删掉重复模板

## License

当前仓库还没有明确 License。

仅供娱乐，不做为实际维修依据