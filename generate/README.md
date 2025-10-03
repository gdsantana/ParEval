# Generate

This subdirectory contains scripts for generating the LLM outputs for ParEval.
The main script is `generate.py`. It can be run as follows.

```sh
usage: generate.py [-h] --prompts PROMPTS --model MODEL --output OUTPUT [--restart] [--cache CACHE] [--restore_from RESTORE_FROM] [--max_new_tokens MAX_NEW_TOKENS]
                   [--num_samples_per_prompt NUM_SAMPLES_PER_PROMPT] [--temperature TEMPERATURE] [--top_p TOP_P] [--do_sample] [--batch_size BATCH_SIZE] [--prompted]
                   [--hf_token HF_TOKEN] [--device DEVICE] [--low_memory] [--dtype {float16,bfloat16,float32}]

Generate code

optional arguments:
  -h, --help            show this help message and exit
  --prompts PROMPTS     Path to the prompt JSON file
  --model MODEL         Path to the language model
  --output OUTPUT       Path to the output JSON file
  --restart             Restart generation from scratch (default: False)
  --cache CACHE         JSONL file to cache intermediate results in. Will be restored from if it already exists and --restart is not specified
  --restore_from RESTORE_FROM
                        JSON file to restore old results from. Will be restored from if it already exists and --restart is not specified. Is different from --cache in that it is a JSON
                        file, not a JSONL file, and it is only used to restore old results where the prompt is equivalent. Cached results are prioritized over restored results.
  --max_new_tokens MAX_NEW_TOKENS
                        Maximum number of new tokens to generate (default: 1024)
  --num_samples_per_prompt NUM_SAMPLES_PER_PROMPT
                        Number of code samples to generate (default: 50)
  --temperature TEMPERATURE
                        Temperature for controlling randomness (default: 0.2)
  --top_p TOP_P         Top p value for nucleus sampling (default: 0.95)
  --do_sample           Enable sampling (default: False)
  --batch_size BATCH_SIZE
                        Batch size for generation (default: 8)
  --prompted            Use prompted generation. See StarCoder paper (default: False)
  --hf_token HF_TOKEN   HuggingFace API token for loading models
  --device DEVICE       Device to use: "cpu", "cuda", "auto" (default: auto)
  --low_memory          Enable low memory mode (reduces batch size and uses CPU offloading)
  --dtype {float16,bfloat16,float32}
                        Override data type for model weights
```

To get outputs in a reasonable amount of time you will most likely need to run
this script on a system with a GPU available. However, if you encounter CUDA
out of memory errors, you can use the new memory optimization options described below.
There are the `generate-openai.py` and `generate-gemini.py` scripts for generating 
outputs from the respective APIs. These will require you to set the API keys in 
environment variables or pass them as arguments. Use `--help` to see the arguments available.

The `--model` argument in `generate.py` can either be a local model path or the
handle of a HuggingFace model. The cache and restart arguments can be used to
restart from existing outputs. The `--prompted` flag will append _solution_
comments to the front of the prompt as in the StarCoder paper. You likely want
this turned on as it almost always improves the results.

## Memory Optimization

If you encounter CUDA out of memory errors, you can use the following options:

### Run on CPU
Use `--device cpu` to run the model entirely on CPU using system RAM instead of GPU VRAM:
```sh
python generate.py --prompts prompts.json --model deepseek-ai/deepseek-coder-6.7b-base --output output.json --device cpu
```

### Low Memory Mode
Use `--low_memory` to enable memory optimizations (reduces batch size and enables CPU offloading):
```sh
python generate.py --prompts prompts.json --model deepseek-ai/deepseek-coder-6.7b-base --output output.json --low_memory
```

### Data Type Optimization
Use `--dtype` to control memory usage:
- `float32`: Most compatible, especially for CPU (higher memory usage)
- `float16`: Reduced memory usage (GPU only)
- `bfloat16`: Balanced option (GPU only)

### Combined Example
For systems with limited VRAM, combine multiple options:
```sh
python generate.py --prompts prompts.json --model deepseek-ai/deepseek-coder-6.7b-base --output output.json --device cpu --low_memory --dtype float32 --batch_size 2
```

## Adding New LLMs

Since a number of the LLMs have different inference settings and prompt formats,
we utilize an inference configuration abstraction to set these parameters for
different models. These are defined in `utils.py`. To add support for a new LLM
you will need to define an inference config for that LLM in `utils.py`. The
existing examples in that file should suffice as examples.

## Translate Tests

Each of the generate scripts has a translation equivalent. These are used for
the translation tasks.