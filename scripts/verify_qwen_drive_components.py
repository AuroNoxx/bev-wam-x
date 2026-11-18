"""User-facing entry point for the isolated v5 component audit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v5_qwen_drive_world.verify_components import main

if __name__ == "__main__":
    main()
