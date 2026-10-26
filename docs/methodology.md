# Methodology and provenance

## Historical record

The curated dataset comes from four measurement sessions on 2026-07-25, associated with an AI infrastructure study by Andrés Rubio and Joel Morales on CEDIA's NVIDIA A100-SXM4-40GB resources. Only aggregate numeric concurrency results from selected runs are included here. The source commit is `1d9c9673d7f46df4f39778231e31ec44744d831f`; each extracted series lists its source path and SHA-256 in `results/a100-concurrency.json`.

The record includes an Ollama run of `qwen3-coder:30b` and two vLLM runs of `qwen25-coder-32b-awq`, one with an AWQ Marlin configuration. Different model and quantization choices prevent attributing all throughput differences to the serving framework. Each concurrency point was a finite batch, not a sustained-load trial with confidence intervals. Prompt/token counts, early EOS, cache residency and kernel choices matter. These are historical observations, not current hardware recommendations.

Private hostnames, endpoint origins, prompt and response text, operational logs, institutional proposals and source participant data are excluded. Numerical values are retained without silently reinterpreting the original metric definitions. In particular, the historical `aggregate_tokens_per_s` is total completed output tokens divided by batch wall time. The historical first-token field is retained as a mean, not relabelled as a percentile.

## New portable client

`bench.py` is an independent public implementation prepared in October 2026. It uses streaming NDJSON for [Ollama](https://docs.ollama.com/api/generate) and SSE for [vLLM's OpenAI-compatible API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/). Its unit tests use synthetic fixtures and real loopback HTTP. The July measurements were not rerun with this client, and synthetic test results are not plotted as GPU measurements.

Before comparing runs, record GPU model/partition, driver, CUDA and framework versions, exact weights and tokenizer, quantization/kernel, input length and output cap, server cache policy, context capacity, warmup procedure, concurrency and competing load. A fixed output cap does not guarantee equal generated lengths. Repeat measurements and report variability before drawing general performance conclusions.

Power is sampled through `nvidia-smi`. Account for the sample interval, idle baseline and other work on the device before integrating energy. It does not measure complete server or wall-plug power.

## Recreating the figure

```bash
python3 -m pip install matplotlib
python3 plot_results.py
```

This plots the checked-in historical summary only. Source code is MIT-licensed. The numerical record is published with attribution to the original campaign and compute provider; it does not reproduce institutional documents, third-party model weights or private workload text.
