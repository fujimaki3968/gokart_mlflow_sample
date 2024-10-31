import logging
import math
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import luigi
import mlflow
import pandas as pd
import pytrec_eval
from gokart.parameter import TaskInstanceParameter
from gokart.target import TargetOnKart

from src.datasets.sushi_dataset import SushiReadExperimentControlFile
from src.mlflow_gokart import MlflowTask

logger = logging.getLogger(__name__)


# 型定義
SearchResultType = dict[str, Any]
SearchResultsType = list[SearchResultType]


class SushiGenerateSearchResults(MlflowTask, ABC):
    """Generate search results for Sushi task A."""

    ecf = TaskInstanceParameter(expected_type=SushiReadExperimentControlFile)

    def requires(self):  # type: ignore  # noqa: ANN201
        """Require."""
        return self.ecf

    def output(self) -> TargetOnKart:
        """Output."""
        return self.make_target("sushi_generate_search_results.pkl")

    @property
    def parameters(self) -> dict:
        """Parameters."""
        return {
            "ecf": self.load(),
        }

    @classmethod
    @abstractmethod
    def _run(cls, ecf: dict, *params: Any, **kwargs: Any) -> SearchResultsType:  # noqa: ANN401
        """Run."""
        raise NotImplementedError


class SushiWriteSearchResults(MlflowTask):
    """Write search results to a file."""

    results = TaskInstanceParameter(expected_type=SushiGenerateSearchResults)
    run_name = luigi.Parameter()

    def requires(self):  # type: ignore  # noqa: ANN201
        """Require."""
        return self.results

    def output(self) -> TargetOnKart:
        """Output."""
        return self.make_target("sushi_write_search_results.tsv")

    @property
    def parameters(self) -> dict:
        """Parameters."""
        return {
            "results": self.load(),
            "run_name": self.run_name,
        }

    @classmethod
    def _run(cls, results: SearchResultsType, run_name: str) -> pd.DataFrame:
        """Run."""
        return cls.write_search_results(results, run_name)

    @staticmethod
    def write_search_results(results: SearchResultsType, run_name: str) -> pd.DataFrame:
        """Write search results."""
        data = [
            [topic["Id"], topic["RankedList"][i], i + 1, 1 / (i + 1), run_name]
            for topic in results
            for i in range(len(topic["RankedList"]))
        ]

        return pd.DataFrame(
            data, columns=pd.Index(["Id", "RankedList", "Rank", "Score", "RunName"])
        )


class SushiEvaluateSearchResult(MlflowTask):
    """Evaluate search results on Sushi task A."""

    search_result = TaskInstanceParameter(expected_type=SushiWriteSearchResults)
    folder_qrels_file_path = luigi.Parameter()
    box_qrels_file_path = luigi.Parameter()
    sushi_files_dir = luigi.Parameter(
        default="data/external/sushi/subtaskA/sushi-files/"
    )
    model_params = luigi.DictParameter(default={})

    def requires(self) -> dict:
        """Require."""
        return {
            "search_result": self.search_result,
        }

    def output(self) -> TargetOnKart:
        """Output."""
        return self.make_target("sushi_evaluate_search_results.pkl")

    @property
    def mlflow_run_name(self) -> str:
        """Run name."""
        run_name = self.search_result.run_name  # type: ignore
        return f"{run_name} - {self.make_unique_id()}"

    @property
    def parameters(self) -> dict:
        """Parameters."""
        return {
            "run_file_name": self.search_result.output().path(),  # type: ignore
            "run_name": self.search_result.run_name,  # type: ignore
            "folder_qrels_file_path": self.folder_qrels_file_path,
            "box_qrels_file_path": self.box_qrels_file_path,
            "sushi_files_dir": self.sushi_files_dir,
            "model_params": self.model_params,
        }

    @classmethod
    def _run(  # noqa: PLR0913
        cls,
        run_file_name: str,
        run_name: str,
        folder_qrels_file_path: str,
        box_qrels_file_path: str,
        sushi_files_dir: str,
        model_params: dict,
    ) -> dict:
        """Run."""
        evaluate_result = cls.evaluate_search_results(
            run_file_name, folder_qrels_file_path, box_qrels_file_path, sushi_files_dir
        )
        mlflow.set_tag("run_name", run_name)
        mlflow.log_params(model_params)
        for measure in evaluate_result:
            mlflow.log_metric(
                f"folder_{measure}", evaluate_result[measure]["folder"]["mean"]
            )
            mlflow.log_metric(f"box_{measure}", evaluate_result[measure]["box"]["mean"])
            mlflow.log_metric(
                f"folder_{measure}_conf", evaluate_result[measure]["folder"]["conf"]
            )
            mlflow.log_metric(
                f"box_{measure}_conf", evaluate_result[measure]["box"]["conf"]
            )

        return evaluate_result

    @staticmethod
    def create_folder_to_box_map(directory: str) -> dict:
        """Create a map from folder to box."""
        box_map = {}
        for box in os.listdir(directory):
            for folder in os.listdir(Path(directory) / box):
                if folder in box_map:
                    logger.warning(
                        "Duplicate folder %s in boxes %s and %s",
                        folder,
                        box,
                        box_map[folder],
                    )
                box_map[folder] = box
        return box_map

    @classmethod
    def make_box_run(cls, sushi_files_dir: str, folder_run: dict) -> dict:
        """Convert folder run to box run."""
        box_map = cls.create_folder_to_box_map(sushi_files_dir)
        box_run = {}
        for topic_id in folder_run:
            box_run[topic_id] = {}
            for folder in folder_run[topic_id]:
                if box_map[folder] not in box_run[topic_id]:
                    box_run[topic_id][box_map[folder]] = folder_run[topic_id][folder]
        return box_run

    @staticmethod
    def stats(results: dict, measure: str) -> tuple[float, float]:
        """Calculate mean and confidence interval of a measure."""
        sum_score = 0
        squaredev = 0
        n = len(results)
        for topic in results:
            sum_score += results[topic][measure]
        mean = sum_score / n
        for topic in results:
            squaredev += (results[topic][measure] - mean) ** 2
        variance = squaredev / (n - 1)
        conf = 1.96 * math.sqrt(variance) / math.sqrt(n)
        return mean, conf

    @classmethod
    def evaluate_search_results(
        cls,
        run_file_name: str,
        folder_qrels_file_path: str,
        box_qrels_file_path: str,
        sushi_files_dir: str,
    ) -> dict:
        """Evaluate search results."""
        evaluate_results = defaultdict(dict)

        #    print(pytrec_eval.supported_measures)
        measures = {
            "ndcg_cut",
            "map",
            "recip_rank",
            "success",
        }  # Generic measures for configuring a pytrec_eval evaluator
        measure_names = {
            "ndcg_cut_5": "NDCG@5",
            "map": "   MAP",
            "recip_rank": "   MRR",
            "success_1": "   S@1",
        }  # Spedific measures for printing in pytrec_eval results

        with (
            Path(run_file_name).open() as run_file,
            Path(folder_qrels_file_path).open() as folder_qrels_file,
            Path(box_qrels_file_path).open() as box_qrels_file,
        ):
            folder_run = {}
            # run_fileの一行目はヘッダーなのでスキップ
            next(run_file)
            for line in run_file:
                topic_id, folder_id, rank, score, run_name = line.split("\t")
                if topic_id not in folder_run:
                    folder_run[topic_id] = {}
                folder_run[topic_id][folder_id] = float(score)
            box_run = cls.make_box_run(sushi_files_dir, folder_run)
            folder_qrels = {}
            for line in folder_qrels_file:
                topic_id, unused, folder_id, relevance_level = line.split("\t")
                if topic_id not in folder_qrels:
                    folder_qrels[topic_id] = {}
                folder_qrels[topic_id][folder_id] = int(
                    relevance_level.strip()
                )  # this deletes the \n at end of line
            folder_evaluator = pytrec_eval.RelevanceEvaluator(folder_qrels, measures)
            folder_topic_results = folder_evaluator.evaluate(
                folder_run
            )  # replace run with folderQrels to see perfect evaluation measures

            box_qrels = {}
            for line in box_qrels_file:
                topic_id, unused, folder_id, relevance_level = line.split("\t")
                if topic_id not in box_qrels:
                    box_qrels[topic_id] = {}
                if folder_id in box_qrels[topic_id]:
                    box_qrels[topic_id][folder_id] = max(
                        box_qrels[topic_id][folder_id], int(relevance_level.strip())
                    )  # strip() deletes the \n at end of line
                else:
                    box_qrels[topic_id][folder_id] = int(relevance_level.strip())
            box_evaluator = pytrec_eval.RelevanceEvaluator(box_qrels, measures)
            box_topic_results = box_evaluator.evaluate(
                box_run
            )  # replace run with qrels to see perfect evaluation measures

            pm = "\u00b1"
            logger.info("          Folder          Box")
            for measure in measure_names:
                folder_mean, folder_conf = cls.stats(folder_topic_results, measure)
                box_mean, box_conf = cls.stats(box_topic_results, measure)

                logger.info(
                    f"{measure_names[measure]}: {folder_mean:.3f}{pm}{folder_conf: .2f}    {box_mean:.3f}{pm}{box_conf: .2f}"  # noqa: E501, G004
                )

                evaluate_results[measure] = {
                    "folder": {"mean": folder_mean, "conf": folder_conf},
                    "box": {"mean": box_mean, "conf": box_conf},
                }

        return evaluate_results
