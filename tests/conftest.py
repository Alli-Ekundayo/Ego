import sys
from unittest.mock import MagicMock

# Mock turbovec to avoid cblas_sgemm ImportError on systems without openblas
sys.modules["turbovec"] = MagicMock()
