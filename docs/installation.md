# 安装与常见问题

## Windows 环境检查

使用 Python 3.10，在 PowerShell 中先检查：

```powershell
py -3.10 --version
```

如果没有 `py` 启动器，但 Python 已加入 PATH，可以使用 `python --version`。安装脚本依次查找 `py -3.10` 和 PATH 中的 Python，并验证版本。也可指定完整路径：

```powershell
.\scripts\setup.ps1 -PythonExecutable 'C:\Python310\python.exe'
```

`-CheckOnly` 只检查 Python，不创建环境或安装依赖。已有 `.venv` 必须使用 Python 3.10；脚本不会覆盖不兼容的环境。

## 安装与启动

```powershell
.\scripts\setup.ps1
.\scripts\run.ps1
```

默认安装关键词检索和聊天界面需要的依赖。打开 `http://127.0.0.1:8501`，选择 BM25 即可检索随仓库提供的资料；不需要聊天 API 或模型权重。

要使用 BGE 或混合检索，安装额外的本地模型依赖并构建索引：

```powershell
.\scripts\setup.ps1 -WithBge
.\.venv\Scripts\python.exe -X utf8 build_bge_combined_index.py
```

首次构建需要下载模型。正常检索只使用本地缓存；缓存或索引缺失时显示关键词降级。

## PowerShell 阻止脚本

如果出现“系统禁止运行脚本”，可仅为当前命令允许执行项目脚本，无需修改系统级执行策略：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

在受组织策略管理的设备上，应按组织要求运行。

## 依赖或模型下载失败

- **找不到 Python 3.10**：检查上面的版本命令，或用 `-PythonExecutable` 指定路径。
- **pip 连接、证书或代理错误**：检查终端中 pip 的具体错误与网络配置。安装失败会保留环境，修复后重新运行安装脚本。
- **No matching distribution found**：确认使用 Python 3.10 和 64 位解释器；检查软件源是否提供锁定版本。
- **BGE 权重下载失败**：先使用 BM25；网络可用后重新构建索引。
- **聊天入口不可用**：在 `.env` 中配置 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`，然后重启应用。
- **8501 端口被占用**：停止已有实例，或使用下面的命令更换端口。

```powershell
.\.venv\Scripts\python.exe -m streamlit run UI.py --server.address 127.0.0.1 --server.port 8502
.\.venv\Scripts\python.exe -X utf8 scripts/doctor.py --json
```

诊断命令只报告配置是否存在，不输出密钥。

## Linux/macOS

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run UI.py --server.address 127.0.0.1
```

需要 BGE 时安装 `requirements-bge.txt`，再运行 `build_bge_combined_index.py`。完整安装尚未在全新 Windows 和 macOS 环境逐项验证。
