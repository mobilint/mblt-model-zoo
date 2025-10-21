from mblt_model_zoo.transformers import pipeline

model_name = "mobilint/bert-base-uncased"

pipe = pipeline('fill-mask', model=model_name)

output = pipe("Hello I'm a [MASK] model.")
print(output)