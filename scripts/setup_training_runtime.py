"""Copy the validated ROCm runtime without modifying its source installation."""
from pathlib import Path
import shutil
import subprocess
import json

root=Path(__file__).resolve().parents[1]
source_python=Path(r'D:\ai\tools\uv-python\cpython-3.13-windows-x86_64-none')
local_python=root/'runtime/python-train/cpython-3.13'
if not local_python.exists():
    print('Copy project-local Python runtime',flush=True)
    shutil.copytree(source_python,local_python)
venv=root/'.venv-train'
if not venv.exists():
    subprocess.run(['uv','venv','--python',str(local_python/'python.exe'),str(venv)],check=True)
source=Path(r'E:\ai\projects\Snap3D-Coder-4B\.venv\Lib\site-packages')
target=venv/'Lib/site-packages'
selected=[]
for path in source.iterdir():
    if path.name.startswith(('torch','functorch','amd_torch','rocm','_rocm')):
        print('Copy',path.name,flush=True)
        if path.is_dir():
            shutil.copytree(path,target/path.name,dirs_exist_ok=True)
        else:
            shutil.copy2(path,target/path.name)
        selected.append(path.name)
(root/'logs/training-runtime-copy.json').write_text(json.dumps({'source':str(source),'python_source':str(source_python),'copied':selected},indent=2))
