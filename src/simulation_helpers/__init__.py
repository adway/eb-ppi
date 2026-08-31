"""Task functions submitted to local workers or Submitit."""

from .runner import run_one, run_task_chunk

__all__ = ["run_one", "run_task_chunk"]
