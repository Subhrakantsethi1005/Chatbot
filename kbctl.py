"""Run the knowledge-base pipeline CLI: ``python kbctl.py --help``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kb_pipeline.cli import main  # noqa: E402

raise SystemExit(main())
