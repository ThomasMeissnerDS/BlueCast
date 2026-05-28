import sys
import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["bluecast/tests/test_orchestrator.py::TestStepBuildLoop::test_build_loop_fast_mode", "-v", "-s"]))
