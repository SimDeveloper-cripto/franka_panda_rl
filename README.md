# Overview

```powershell
# Windows11
# 1. dependency needed:python3.10-dev
python -m venv .venv
.\.venv\Scripts\activate 
pip install --no-deps -r requirements.txt 
pip install -r requirements.txt
# 2. copy mujoco.dll into venv\Lib\robosuite\utils 
& .venv\Scripts\python.exe close_generalized/train_gen.py --play
```

```bash
# Linux
# 1. dependency needed:python3.10-dev
python -m venv .venv
source ./.venv/bin/activate
pip install --no-deps -r requirements.txt 
pip install -r requirements.txt
# 2. copy mujoco.dll into venv\Lib\robosuite\utils 
python close_generalized/train_gen.py --play
```

```bash
# macOS
# 1. dependency needed:python3.10-dev
python -m venv .venv
source ./.venv/bin/activate
pip install --no-deps -r requirements.txt 
pip install -r requirements.txt
mjpython close_generalized/train_gen.py --play
```

## Results

Coming soon with v2 !!

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.