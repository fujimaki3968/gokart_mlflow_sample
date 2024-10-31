import logging
import re
import shutil
from pathlib import Path

import luigi
import pandas as pd
import PyPDF2
import pyterrier as pt

from src.evaluation import SushiGenerateSearchResults

logger = logging.getLogger(__name__)


class SushiPyterrierModelSearchResults(SushiGenerateSearchResults):
    """Generate search results for Sushi task A using a PyTerrier model."""

    search_fields = luigi.ListParameter(default=["title", "ocr", "folderlabel"])
    sushi_files_dir = luigi.Parameter(
        default="data/external/sushi/subtaskA/sushi-files/"
    )
    metadata_file_path = luigi.Parameter()
    snc_translation_file_path = luigi.Parameter()

    @property
    def parameters(self) -> dict:
        """Parameters."""
        output_path = Path(self.output().path())
        index_dir = output_path.parent / "index"
        index_dir.mkdir(parents=True, exist_ok=True)

        return {
            "ecf": self.load(),
            "search_fields": self.search_fields,
            "sushi_files_dir": self.sushi_files_dir,
            "metadata_file_path": self.metadata_file_path,
            "snc_translation_file_path": self.snc_translation_file_path,
            "index_dir": str(index_dir),
        }

    @classmethod
    def _run(  # noqa: PLR0913
        cls,
        ecf: dict,
        search_fields: list[str],
        sushi_files_dir: str,
        metadata_file_path: str,
        snc_translation_file_path: str,
        index_dir: str,
    ) -> list[dict]:
        """Run."""
        results = []
        i = 0
        for experiment_set in ecf["ExperimentSets"]:
            index = cls.train_terrier_model(
                experiment_set["TrainingDocuments"],
                search_fields,
                sushi_files_dir,
                metadata_file_path,
                snc_translation_file_path,
                index_dir,
            )
            topics = list(experiment_set["Topics"].keys())
            for j in range(len(topics)):
                results.append({})
                results[i]["Id"] = topics[j]
                query = experiment_set["Topics"][topics[j]]["TITLE"]
                ranked_folder_list = cls.search(query, index)
                results[i]["RankedList"] = ranked_folder_list
                i += 1
        return results

    @classmethod
    def train_terrier_model(  # noqa: C901, PLR0912, PLR0913, PLR0915
        cls,
        training_docs: list[str],
        search_fields: list[str],
        sushi_files_dir: str,
        metadata_file_path: str,
        snc_translation_file_path: str,
        index_dir: str,
    ) -> pt.BatchRetrieve:
        """Train a Terrier model."""
        no_short_ocr = False
        training_set = []

        # Read the Sushi Medadata and SNC excel files
        try:
            file_metadata = pd.read_excel(metadata_file_path, sheet_name=0)
            snc_expansion = pd.read_excel(snc_translation_file_path, sheet_name=0)
        except Exception:
            logger.exception("Error reading metadata files")
            raise

        # Build the data structure that Terrier will index
        for training_doc in training_docs:
            # Read the box/folder/file directory structure
            sushi_file = training_doc[
                -10:
            ]  # This extracts the file name and ignores the box and folder labels which we will get from the medatada  # noqa: E501
            file = sushi_file
            folder = str(
                file_metadata.loc[
                    file_metadata["Sushi File"] == sushi_file, "Sushi Folder"
                ].iloc[0]
            )
            box = str(
                file_metadata.loc[
                    file_metadata["Sushi File"] == sushi_file, "Sushi Box"
                ].iloc[0]
            )

            # Construct the best available folder label (either by SNC lookup or by using the folder label text)  # noqa: E501
            brown_label = str(
                file_metadata.loc[
                    file_metadata["Sushi File"] == sushi_file, "Brown Folder Name"
                ].iloc[0]
            )
            nara_label = str(
                file_metadata.loc[
                    file_metadata["Sushi File"] == sushi_file, "NARA Folder Name"
                ].iloc[0]
            )
            if nara_label != "nan":
                start = nara_label.find("(")  # Strip part markings
                if start != -1:
                    nara_label = nara_label[:start]
                nara_label = nara_label.replace(
                    "BRAZ-A0", "BRAZ-A 0"
                )  # Fix formatting error
                nara_label = nara_label.replace(
                    "BRAZ-E0", "BRAZ-E 0"
                )  # Fix formatting error
                nara_label_elements = nara_label.split()
                if len(nara_label_elements) in [3, 4]:
                    if len(nara_label_elements) == 3:  # noqa: PLR2004
                        nara_snc = nara_label_elements[0]
                    else:
                        nara_snc = " ".join(nara_label_elements[0:2])
                    nara_country_code = nara_label_elements[-2]
                    nara_date = nara_label_elements[-1]

                    logger.info(
                        "Parsed %s to %s // %s // %s",
                        nara_label,
                        nara_snc,
                        nara_country_code,
                        nara_date,
                    )
                    if nara_snc in snc_expansion["SNC"].tolist():
                        label1965 = str(
                            snc_expansion.loc[
                                snc_expansion["SNC"] == nara_snc, 1965
                            ].iloc[0]  # type: ignore
                        )
                        label1963 = str(
                            snc_expansion.loc[
                                snc_expansion["SNC"] == nara_snc, 1963
                            ].iloc[0]  # type: ignore
                        )
                        if label1965 != "nan":
                            label = label1965
                        elif label1963 != "nan":
                            label = label1963
                        else:
                            logger.error(
                                "Unable to translate %s for file %s in folder",
                                nara_snc,
                                sushi_file,
                            )
                            label = nara_snc
                    else:
                        logger.error(
                            "No expansion for %s for file %s in folder",
                            nara_snc,
                            sushi_file,
                        )
                        label = nara_snc

                else:
                    logger.error(
                        "NARA Folder Title doesn't have four parts: %s", nara_label
                    )
                    label = "Bad NARA Folder Title"
            elif brown_label != "nan":
                label = brown_label.replace("_", " ")
            else:
                logger.error(
                    "Missing both NARA and Brown folder labels for file %s in folder %s",  # noqa: E501
                    sushi_file,
                    folder,
                )
                label = "No NARA or Brown Folder Title"
            #        print(f'File {file} Folder {folder} has expanded label {label}')

            # Construct the best available title (either Brown, or trimmed NARA)
            brown_title = str(
                file_metadata.loc[
                    file_metadata["Sushi File"] == sushi_file, "Brown Title"
                ].iloc[0]
            )
            nara_title = str(
                file_metadata.loc[
                    file_metadata["Sushi File"] == sushi_file, "NARA Title"
                ].iloc[0]
            )
            if brown_title != "nan":
                title = brown_title
            else:
                start = nara_title.find("Concerning")
                if start != -1:
                    nara_title = nara_title[start + 11 :]
                end1 = nara_title.rfind(":")
                end2 = nara_title.rfind("(")
                end = min(end1, end2)
                if end != -1:
                    nara_title = nara_title[:end]
                title = nara_title

            # Extract OCR text from the PDF file
            file_path = Path(sushi_files_dir) / box / folder / file
            with file_path.open("rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = len(reader.pages)
                max_pages = (
                    1  # Increase this number if you want to index more of the OCR text
                )
                fulltext = ""
                for i in range(min(pages, max_pages)):
                    page = reader.pages[i]
                    text = page.extract_text().replace("\n", " ")
                    fulltext = fulltext + text
            ocr = fulltext

            # Optionally replace any hopelessly short OCR with the document title
            if no_short_ocr and len(ocr) < 5:  # noqa: PLR2004
                logger.info(
                    "Replaced OCR: //%s// with Title //%s// for file %s in folder %s",
                    ocr,
                    title,
                    file,
                    folder,
                )
                ocr = title

            training_set.append(
                {
                    "docno": file,
                    "folder": folder,
                    "box": box,
                    "title": title,
                    "ocr": ocr,
                    "folderlabel": label,
                }
            )

        # Create the Terrier index for this training set and then return a Terrier retriever for that index  # noqa: E501

        if "index" in index_dir and Path(index_dir).is_dir():
            logger.info("Deleting prior index %s", index_dir)
            shutil.rmtree(
                index_dir
            )  # This is required because Terrier fails to close its index on completion

        if not pt.started():
            pt.init()  # type: ignore

        indexer = pt.IterDictIndexer(
            index_dir,
            meta={
                "docno": 20,
                "folder": 20,
                "box": 20,
                "title": 16384,
                "ocr": 16384,
                "folderlabel": 1024,
            },
            meta_reverse=["docno", "folder", "box"],
            overwrite=True,
        )
        indexref = indexer.index(training_set, fields=search_fields)  # type: ignore
        index = pt.IndexFactory.of(indexref)  # type: ignore
        return pt.BatchRetrieve(
            index, wmodel="BM25", metadata=["docno", "folder", "box"], num_results=1000
        )

    @staticmethod
    def search(query: str, engine: pt.BatchRetrieve) -> list[str]:
        """Search for the query in the index."""
        if not pt.started():
            pt.init()  # type: ignore

        query = re.sub(
            r"[^a-zA-Z0-9\s]", "", query
        )  # Terries fails if punctuation is found in a query
        result = engine.search(query)
        ranked_list = result["folder"]
        ranked_list = ranked_list.drop_duplicates()
        return ranked_list.tolist()
