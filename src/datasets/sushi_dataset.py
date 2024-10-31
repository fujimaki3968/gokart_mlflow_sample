import json
import logging
from pathlib import Path

import luigi

from src.mlflow_gokart import MlflowTask

logger = logging.getLogger(__name__)


class SushiReadExperimentControlFile(MlflowTask):
    """Read an experiment control file for Sushi task A."""

    file_path = luigi.Parameter()

    @property
    def parameters(self) -> dict:
        """Parameters."""
        return {
            "file_path": self.file_path,
        }

    @classmethod
    def _run(cls, file_path: str) -> dict:
        with Path(file_path).open() as ecf_file:
            return json.load(ecf_file)
