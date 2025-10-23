from pprint import pprint
from mblt_model_zoo.transformers import list_models, list_tasks


def test_transformers_api():
    print(list_tasks())
    pprint(list_models())
