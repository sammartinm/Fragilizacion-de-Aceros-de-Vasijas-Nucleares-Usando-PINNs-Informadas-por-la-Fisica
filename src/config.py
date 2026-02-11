from pathlib import Path

def get_project_root() -> Path:
    """Retorna la raíz del proyecto buscando el archivo setup.py"""
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / 'setup.py').exists():
            return parent
    return current_path

# Variables globales disponibles para todo el proyecto
ROOT_DIR = get_project_root()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"