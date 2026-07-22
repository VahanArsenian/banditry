"""Centralised, colourful logging.

Everything user-facing flows through one `rich.console.Console`. Modules
import the helpers below; verbose surrogate/sampler/internal chatter is
gated behind `set_verbose(True)` and otherwise silenced.
"""

from __future__ import annotations

from rich.console import Console


console = Console(highlight=False)
_VERBOSE = False


def set_verbose(flag: bool) -> None:
    """Enable/disable the noisy internal prints (surrogate training,
    SGLD step sizes, GA termination notices, etc.)."""
    
    global _VERBOSE
    _VERBOSE = bool(flag)


def debug(*objects, **kwargs) -> None:
    """Print only when verbose mode is on. Use for surrogate / sampler /
    GA chatter that isn't useful by default."""
    if _VERBOSE:
        console.print(*objects, style="dim", **kwargs)
