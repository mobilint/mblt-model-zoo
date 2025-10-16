# Mobilint Model Zoo

<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="https://raw.githubusercontent.com/mobilint/mblt-model-zoo/master/assets/Mobilint_Logo_Primary.png" alt="Mobilint Logo" width="60%">
</a>
</p>
</div>

**mblt-model-zoo** is a curated collection of AI models optimized by [Mobilint](https://www.mobilint.com/)’s Neural Processing Units (NPUs).

Designed to help developers accelerate deployment, Mobilint's Model Zoo offers access to public, pre-trained, and pre-quantized models for vision, language, and multimodal tasks. Along with performance results, we provide pre- and post-processing tools to help developers evaluate, fine-tune, and integrate the models with ease.

## Installation

- Install Mobilint ACCELerator(MACCEL) on your environment. In case you are not Mobilint customer, please contact [us](mailto:tech-support@mobilint.com).
- Install **mblt-model-zoo** using pip:

```bash
pip install mblt-model-zoo
```

- If you want to install the latest version from the source, clone the repository and install it:

```bash
git clone https://github.com/mobilint/mblt-model-zoo.git
cd mblt-model-zoo
pip install -e .
```

## Quick Start Guide

### Initializing Quantized Model Class

**mblt-model-zoo** provides a quantized model with associated pre- and post-processing tools. The following code snippet shows how to use the pre-trained model for inference.

```python
from mblt_model_zoo.vision import ResNet50

# Load the pre-trained model. 
# Automatically download the model if not found in the local cache.
resnet50 = ResNet50() 

# Load the model trained with different recipe
# Currently, default is "DEFAULT", or "IMAGENET1K_V1.
resnet50 = ResNet50(model_type = "IMAGENET1K_V2")

# Download the model to local directory and load it
resnet50 = ResNet50(local_path = "path/to/local/") # the file will be downloaded to "path/to/local/model.mxq"

# Load the model from a local path or download as filename and file path you want
resnet50 = ResNet50(local_path = "path/to/local/model.mxq")

# Set inference mode for better performance
# Aries supports "single", "multi" and "global" inferece mode. Default is "global"
resnet50 = ResNet50(infer_mode = "global")

# (Beta) If you are holding a model compiled for Regulus, enable inference on the Regulus device.
resnet50 = ResNet50(product = "regulus")

# In summary, the model can be loaded with the following arguments. 
# You may customize those arguments to work with Mobilint's NPU.
resnet50 = ResNet50(
    local_path = None,
    model_type = "DEFAULT",
    infer_mode = "global",
    product = "aries",
)

```

### Working with Quantized Model

With the image given as path, PIL image, numpy array, or torch tensor, you can perform inference with the quantized model. The following code snippet shows how to use the quantized model for inference:

```python
image_path = "path/to/image.jpg"

input_img = resnet50.preprocess(image_path) # Preprocess the input image
output = resnet50(input_img) # Perform inference with the quantized model
result = resnet50.postprocess(output) # Postprocess the output

result.plot(
    source_path=image_path,
    save_path="path/to/save/result.jpg",
)
```

### Listing Available Models

**mblt-model-zoo** offers a function to list all available models. You can use the following code snippet to list the models for a specific task (e.g., image classification, object detection, etc.):

```python
from mblt_model_zoo.vision import list_models
from pprint import pprint

available_models = list_models()
pprint(available_models)
```

## Model List

We provide the models that are quantized with our advanced quantization techniques. List of available vision models are [here](mblt_model_zoo/vision/README.md).

## Optional Extras

When working with tasks other than vision, extra dependencies may be required. Those options can be installed via `pip install mblt-model-zoo[NAME]` or `pip install -e .[NAME]`.

Currently, this optional functions are only available on environment equipped with Mobilint's [Aries](https://www.mobilint.com/aries).

|Name|Use|Details|
|-------|------|------|
|transformers|For using HuggingFace transformers related models|[README.md](mblt_model_zoo/transformers/README.md) |

## License

The Mobilint Model Zoo is released under BSD 3-Clause License. Please see the [LICENSE](https://github.com/mobilint/mblt-model-zoo/blob/master/LICENSE) file for more details.

Additionally, the license for each model provided in this package follows the terms specified in the source link provided with it.

## Support & Issues

If you encounter any problems with this package, please feel free to contact [us](mailto:tech-support@mobilint.com).
