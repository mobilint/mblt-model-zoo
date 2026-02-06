"""
Test script for Vision API.

This script tests the API for listing available models and tasks in the vision module.
"""

from pprint import pprint

from mblt_model_zoo.vision import list_models, list_tasks

print(list_tasks())
pprint(list_models())
