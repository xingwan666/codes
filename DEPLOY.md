# Ubuntu Deployment

这份文档面向 Ubuntu 20.04 服务器部署。

适用场景：
- 低并发
- 偶发上传识别
- 1 核 1G 小服务器

不适合：
- 高频并发调用
- 秒级稳定响应要求很高的正式生产流量

## 1. 机器建议

当前项目依赖 `EasyOCR + torch + CPU`。

在 `1C1G` 机器上可以运行，但建议至少：
- 开 `2G swap`
- `gunicorn` 只开 `1 worker`
- 通过 `nginx` 反向代理

如果后续访问量增大，更推荐：
- `2C4G` 或以上

## 2. 安装系统依赖

```bash
sudo apt update
sudo apt install -y software-properties-common nginx git libgl1 libglib2.0-0
```

## 3. 安装 Python

Ubuntu 20.04 自带 Python 版本通常偏旧，建议使用 `3.10` 或 `3.11`。

示例：

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-distutils
```

## 4. 拉代码

```bash
sudo mkdir -p /opt/tire-ocr
sudo chown -R $USER:$USER /opt/tire-ocr
git clone <your-repo-url> /opt/tire-ocr
cd /opt/tire-ocr
```

## 5. 创建虚拟环境

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 6. 增加 swap

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

## 7. 本机验证

```bash
source .venv/bin/activate
python app.py
```

浏览器访问：

```txt
http://server-ip:5000
```

如果只是临时测试，跑到这一步就够了。

## 8. 使用 gunicorn

```bash
source .venv/bin/activate
gunicorn -w 1 -b 127.0.0.1:5000 --timeout 120 app:app
```

说明：
- `-w 1`：1G 小机器只建议 1 个 worker
- `--timeout 120`：OCR 推理比普通 Web 请求慢，超时要放宽

## 9. 配置 systemd

新建：

```bash
sudo nano /etc/systemd/system/tire-ocr.service
```

内容：

```ini
[Unit]
Description=Tire OCR Flask App
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/tire-ocr
Environment="PATH=/opt/tire-ocr/.venv/bin"
ExecStart=/opt/tire-ocr/.venv/bin/gunicorn -w 1 -b 127.0.0.1:5000 --timeout 120 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable tire-ocr
sudo systemctl start tire-ocr
sudo systemctl status tire-ocr
```

查看日志：

```bash
sudo journalctl -u tire-ocr -f
```

## 10. 配置 nginx

新建：

```bash
sudo nano /etc/nginx/sites-available/tire-ocr
```

内容：

```nginx
server {
    listen 80;
    server_name _;

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

启用：

```bash
sudo ln -sf /etc/nginx/sites-available/tire-ocr /etc/nginx/sites-enabled/tire-ocr
sudo nginx -t
sudo systemctl reload nginx
```

## 11. 更新代码

后续更新建议统一走 Git：

```bash
cd /opt/tire-ocr
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart tire-ocr
```

## 12. 已知限制

- 小内存机器上首次加载 OCR 模型会比较慢
- CPU 推理速度有限
- 高并发时会排队
- 当前版本更适合内部工具，不适合高流量正式生产环境

## 13. 建议的下一步优化

- 增加离线测试脚本，自动统计样图识别效果
- 继续把 OCR 逻辑做成更贴合固定界面的专用识别器
- 如果长期使用，考虑升级服务器规格
