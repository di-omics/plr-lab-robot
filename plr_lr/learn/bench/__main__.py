"""``python -m plr_lr.learn.bench``.

The exit status is the verdict. Zero means the run produced a scorecard that this
harness is willing to stand behind; anything else means it produced a refusal, and a
refusal is a different outcome from a low score rather than a worse one.
"""

from __future__ import annotations

import sys

from .commands import main

if __name__ == "__main__":
  sys.exit(main())
