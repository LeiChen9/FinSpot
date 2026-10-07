"""项目根目录与本地数据路径(数据文件一律不入 git)。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIN_DIR = DATA_DIR / "financial"
DIV_DIR = DATA_DIR / "dividend"
META_DIR = DATA_DIR / "meta"
