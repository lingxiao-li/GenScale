# Inference Code Bundle

This folder contains anonymized copies of the inference entry points and the local
project modules they import.

Bundled entry points:

- `inference_size_correction.py`
- `inference_size_correction_t2.py`
- `inference_size_correction_t3.py`
- `inference_size_correction.sh`

Local dependencies are included under `src/models/`, plus `multi_object_inference.py`.

Runtime model checkpoints are not bundled. Configure them with environment
variables or command-line arguments:

- `INSERTANYTHING_WEIGHTS_DIR`: LoRA + ControlNet checkpoint directory.
- `SAM2_CHECKPOINT`: SAM2 checkpoint path.
- `DEPTH_ANYTHING_V2_PATH`: optional Depth-Anything-V2 source checkout.
- `DEPTH_ANYTHING_V2_CHECKPOINT`: Depth-Anything-V2 checkpoint path.
- `FLUX_FILL_PATH`, `FLUX_REDUX_PATH`, `FLUX_KONTEXT_PATH`: local paths or
  Hugging Face model ids for FLUX components.

The default benchmark path in Task2/Task3 wrappers points to the anonymized
benchmark JSON in the parent submission directory.

Additional benchmark/evaluation utilities are included under `scripts/`:

- `scripts/generate_benchmark_dataset.py`
- `scripts/eval/task1/evaluate_genscale_task1_gemini_v5.py`
- `scripts/eval/task2/evaluate_genscale_task2_gemini_v5.py`
- `scripts/eval/task3/evaluate_genscale_task3_gemini_v5.py`

The evaluation utilities read `GOOGLE_API_KEY` from the environment. No API keys
are embedded in the source files.
