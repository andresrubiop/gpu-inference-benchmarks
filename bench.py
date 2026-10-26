#!/usr/bin/env python3
"""Measure closed-loop LLM inference over Ollama NDJSON or vLLM/OpenAI SSE."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


def percentile(values, q):
    """Linearly interpolated percentile, or None for an empty population."""
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def consume_stream(lines, backend, started, clock=time.perf_counter):
    first = None
    tokens = None
    terminal = False
    for raw in lines:
        text = raw.decode('utf-8').strip()
        if not text or text.startswith(':'):
            continue
        if backend == 'vllm':
            if not text.startswith('data:'):
                continue
            text = text[5:].strip()
            if text == '[DONE]':
                terminal = True
                break
        event = json.loads(text)
        if event.get('error'):
            raise ValueError('server_error')
        if backend == 'ollama':
            content = event.get('response', '')
            if event.get('done'):
                terminal = True
                tokens = event.get('eval_count')
        else:
            content = ''.join(c.get('text') or c.get('delta', {}).get('content') or ''
                              for c in event.get('choices', []))
            if event.get('usage'):
                tokens = event['usage'].get('completion_tokens')
        if content and first is None:
            first = clock() - started
    if not terminal:
        raise ValueError('incomplete_stream')
    if tokens is not None and (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0):
        raise ValueError('invalid_token_count')
    return {'latency_s': clock() - started, 'first_content_s': first, 'output_tokens': tokens}


def request_sample(backend, base_url, model, prompt, max_tokens, context, timeout):
    if backend == 'ollama':
        endpoint = '/api/generate'
        payload = {'model': model, 'prompt': prompt, 'stream': True,
                   'options': {'num_predict': max_tokens, 'num_ctx': context, 'temperature': 0}}
    else:
        endpoint = '/v1/completions'
        payload = {'model': model, 'prompt': prompt, 'stream': True,
                   'stream_options': {'include_usage': True}, 'max_tokens': max_tokens,
                   'temperature': 0}
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('BENCH_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['BENCH_API_KEY']
    req = urllib.request.Request(base_url.rstrip('/') + endpoint,
                                 data=json.dumps(payload).encode(), headers=headers, method='POST')
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = consume_stream(response, backend, started)
        return {'ok': True, **result}
    except urllib.error.HTTPError as exc:
        return {'ok': False, 'error': 'http_' + str(exc.code)}
    except (OSError, ValueError, UnicodeError, TypeError, KeyError) as exc:
        # Do not persist response bodies, prompt text, endpoint credentials or tokens.
        return {'ok': False, 'error': type(exc).__name__}


def summarize(samples, elapsed):
    good = [s for s in samples if s['ok']]
    latencies = [s['latency_s'] for s in good]
    first = [s['first_content_s'] for s in good if s['first_content_s'] is not None]
    known = [s['output_tokens'] for s in good if s['output_tokens'] is not None]
    # No invented throughput when the server omits token usage on any success.
    count = sum(known) if good and len(known) == len(good) else None
    return {'requests': len(samples), 'succeeded': len(good), 'failed': len(samples) - len(good),
            'wall_s': elapsed, 'output_tokens': count,
            'aggregate_tokens_per_s': count / elapsed if count is not None and elapsed > 0 else None,
            'latency_p50_s': percentile(latencies, .5), 'latency_p95_s': percentile(latencies, .95),
            'first_content_p50_s': percentile(first, .5), 'first_content_p95_s': percentile(first, .95),
            'samples': samples}


def run_level(sample, concurrency, requests):
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        samples = list(executor.map(lambda _: sample(), range(requests)))
    return {'concurrency': concurrency, **summarize(samples, time.perf_counter() - started)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['ollama', 'vllm'], required=True)
    parser.add_argument('--base-url', help='Server origin; default is loopback for the chosen backend')
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt-file', type=Path, default=Path(__file__).parent / 'prompts/example.txt')
    parser.add_argument('--concurrency', default='1,4,8')
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--context', type=int, default=4096, help='Ollama num_ctx; configure vLLM server separately')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--output', type=Path, default=Path('results/run.json'))
    args = parser.parse_args(argv)
    try:
        levels = [int(n) for n in args.concurrency.split(',')]
    except ValueError:
        parser.error('concurrency must be comma-separated positive integers')
    if not levels or any(n < 1 or n > args.requests for n in levels):
        parser.error('each concurrency must be between 1 and requests')
    if args.requests < 1 or args.max_tokens < 1 or args.context < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error('counts, context and timeout must be positive; warmup may be zero')
    base = args.base_url or ('http://127.0.0.1:11434' if args.backend == 'ollama' else 'http://127.0.0.1:8000')
    url = urllib.parse.urlparse(base)
    if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
        parser.error('use an HTTP(S) origin without credentials, query or fragment')
    if url.path not in {'', '/'}:
        parser.error('base-url must be an origin, not an API path')
    prompt = args.prompt_file.read_text()
    sample = lambda: request_sample(args.backend, base, args.model, prompt, args.max_tokens, args.context, args.timeout)
    for _ in range(args.warmup):
        if not sample()['ok']:
            parser.error('warmup failed; verify the server, model and credentials')
    results = [run_level(sample, n, args.requests) for n in levels]
    report = {'schema_version': 1, 'backend': args.backend, 'model': args.model,
              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'max_tokens': args.max_tokens, 'ollama_context': args.context if args.backend == 'ollama' else None,
              'warmup_requests_excluded': args.warmup, 'levels': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for level in results:
        print(json.dumps({k: v for k, v in level.items() if k != 'samples'}))
    return 1 if any(level['failed'] for level in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
