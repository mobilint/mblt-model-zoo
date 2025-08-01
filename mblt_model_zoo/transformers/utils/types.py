from typing import List
from dataclasses import dataclass


@dataclass
class TransformersModelInfo:
    """
    This class is used to store model information from huggingface transformers library.
    """

    original_model_id: str
    model_id: str
    download_url_base: str
    file_list: List[str]

    def __str__(self):
        return f"'{self.model_id}'"

    def __repr__(self):
        return self.__str__()

    def get_directory_name(self):
        return self.model_id[len("mobilint/") :]
