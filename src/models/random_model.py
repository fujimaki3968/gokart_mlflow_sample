import os
import random
from pathlib import Path

import luigi

from src.evaluation import SearchResultsType, SushiGenerateSearchResults


class SushiRandomModelSearchResults(SushiGenerateSearchResults):
    """Generate search results for Sushi task A using a random model."""

    sushi_files_dir = luigi.Parameter(
        default="data/external/sushi/subtaskA/sushi-files/"
    )

    @property
    def parameters(self) -> dict:
        """Parameters."""
        return {
            "ecf": self.load(),
            "sushi_files_dir": self.sushi_files_dir,
        }

    @classmethod
    def _run(cls, ecf: dict, sushi_files_dir: str) -> SearchResultsType:
        """Run."""
        results = []
        i = 0
        for experiment_set in ecf["ExperimentSets"]:
            index = cls.train_random_model(sushi_files_dir)
            topics = list(experiment_set["Topics"].keys())
            for j in range(len(topics)):
                results.append({})
                results[i]["Id"] = topics[j]
                query = experiment_set["Topics"][topics[j]]["TITLE"]
                ranked_folder_list = cls.search(query, index)
                results[i]["RankedList"] = ranked_folder_list
                i += 1
        return results

    @staticmethod
    def search(query: str, index: list[str]) -> list[str]:  # noqa: ARG004
        """Search for the query in the index."""
        return index

    @staticmethod
    def train_random_model(
        sushi_files_dir: str,
    ) -> list[str]:
        """Train a random model.(ここでは検索対象をランダムに選択し返すだけ)"""
        folders = {}
        for box in os.listdir(sushi_files_dir):
            for folder in os.listdir(Path(sushi_files_dir) / box):
                folders[folder] = []
                for file in os.listdir(Path(sushi_files_dir) / box / folder):
                    folders[folder].append(file)
        index = list(folders.keys())
        random.shuffle(index)
        return index[0:1000]
