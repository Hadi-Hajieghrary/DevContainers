#!/usr/bin/env python3
"""Demo 11: render a short recorded nuPlan scenario to MP4.

This demo is the CLI entrypoint for the shared rendering utility in
`demos/common/recorded_scenario_renderer.py`.
"""

from __future__ import annotations

import sys

from common.recorded_scenario_renderer import main


if __name__ == "__main__":
    sys.exit(main())
