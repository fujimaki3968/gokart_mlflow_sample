import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging
from datetime import datetime

import luigi
from gokart.build import build as gokart_build
from gokart.utils import add_config as gokart_add_config

from src.datasets.sushi_dataset import SushiReadExperimentControlFile
from src.evaluation import SushiEvaluateSearchResult, SushiWriteSearchResults
from src.mlflow_gokart import MlflowTask
from src.models.pyterrier_model import SushiPyterrierModelSearchResults
from src.models.random_model import SushiRandomModelSearchResults

logger = logging.getLogger(__name__)


class SushiPipeline(MlflowTask):
    """Sushi pipeline."""

    ecf_file_path = luigi.Parameter()
    sushi_files_dir = luigi.Parameter()
    metadata_file_path = luigi.Parameter()
    snc_translation_file_path = luigi.Parameter()
    folder_qrels_file_path = luigi.Parameter()
    box_qrels_file_path = luigi.Parameter()

    @property
    def parameters(self) -> dict:
        """Parameters."""
        return {}

    def requires(self):  # type: ignore # noqa: ANN201
        """Require."""
        # dataset
        ecf = SushiReadExperimentControlFile(file_path=self.ecf_file_path)

        # model
        search_result_random = SushiWriteSearchResults(
            results=SushiRandomModelSearchResults(
                sushi_files_dir=self.sushi_files_dir,
                ecf=ecf,
            ),
            run_name="random_model",
        )

        eval_random_result = SushiEvaluateSearchResult(
            search_result=search_result_random,
            folder_qrels_file_path=self.folder_qrels_file_path,
            box_qrels_file_path=self.box_qrels_file_path,
            sushi_files_dir=self.sushi_files_dir,
        )

        eval_pyterrier_results = []
        for search_fields in [
            ["title"],
            ["ocr"],
            ["folderlabel"],
            ["title", "ocr"],
            ["title", "folderlabel"],
            ["ocr", "folderlabel"],
            ["title", "ocr", "folderlabel"],
        ]:
            search_result_pyterrier = SushiWriteSearchResults(
                results=SushiPyterrierModelSearchResults(
                    ecf=ecf,
                    search_fields=search_fields,
                    sushi_files_dir=self.sushi_files_dir,
                    metadata_file_path=self.metadata_file_path,
                    snc_translation_file_path=self.snc_translation_file_path,
                ),
                run_name=f"pyterrier_model_{'+'.join(search_fields)}",
            )

            eval_pyterrier_result = SushiEvaluateSearchResult(
                search_result=search_result_pyterrier,
                folder_qrels_file_path=self.folder_qrels_file_path,
                box_qrels_file_path=self.box_qrels_file_path,
                sushi_files_dir=self.sushi_files_dir,
                model_params={"search_fields": search_fields},
            )

            eval_pyterrier_results.append(eval_pyterrier_result)

        return {
            "eval_random_result": eval_random_result,
            "eval_pyterrier_results": eval_pyterrier_results,
        }

    @property
    def mlflow_experiment_name(self) -> str:
        """MLflow の実験名を返す."""
        return "SushiPipeline"

    @property
    def mlflow_run_name(self) -> str:
        """Run name."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ005

    def _run(self) -> str:
        return self.load()


if __name__ == "__main__":
    import pathlib

    # data/external/sushiが存在していない場合はダウンロードshell scriptを実行
    sushi_files_dir = pathlib.Path("data/external/sushi")

    if not sushi_files_dir.exists():
        import subprocess

        if input("Download Sushi dataset? [y/n]: ").lower() == "y":
            subprocess.run(["sh", "src/datasets/download_sushi_task.sh"], check=True)
        else:
            print("Please download the Sushi dataset.")
            sys.exit(1)

    gokart_add_config("src/param.ini")
    gokart_build(SushiPipeline())
