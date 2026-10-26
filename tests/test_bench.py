import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from bench import consume_stream, percentile, request_sample, run_level, summarize


class StreamTests(unittest.TestCase):
    def test_ollama_uses_final_usage_and_first_visible_content(self):
        stream = io.BytesIO(b'{"thinking":"hidden"}\n{"response":"hi"}\n{"done":true,"eval_count":9}\n')
        with patch('bench.time.perf_counter', return_value=5):
            result = consume_stream(stream, 'ollama', 1, clock=lambda: 5)
        self.assertEqual(result['output_tokens'], 9)
        self.assertEqual(result['first_content_s'], 4)

    def test_sse_usage_chunk_can_have_no_choices(self):
        stream = io.BytesIO(b': keepalive\n\ndata: {"choices":[{"text":"hello"}]}\n\ndata: {"choices":[],"usage":{"completion_tokens":7}}\n\ndata: [DONE]\n\n')
        result = consume_stream(stream, 'vllm', 1, clock=lambda: 2)
        self.assertEqual(result['output_tokens'], 7)
        self.assertEqual(result['first_content_s'], 1)

    def test_truncated_stream_is_failure(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            consume_stream(io.BytesIO(b'{"response":"partial"}\n'), 'ollama', 0)

    def test_missing_usage_stays_unknown(self):
        result = consume_stream(io.BytesIO(b'data: [DONE]\n'), 'vllm', 0)
        self.assertIsNone(result['output_tokens'])

    def test_server_error_is_not_a_success(self):
        with self.assertRaises(ValueError):
            consume_stream(io.BytesIO(b'{"error":"failed"}\n'), 'ollama', 0)

    def test_boolean_token_count_is_invalid(self):
        with self.assertRaises(ValueError):
            consume_stream(io.BytesIO(b'{"done":true,"eval_count":true}\n'), 'ollama', 0)


class SummaryTests(unittest.TestCase):
    def test_empty_population_is_not_zero_latency(self):
        self.assertIsNone(percentile([], .5))
        self.assertIsNone(summarize([{'ok': False}], 1)['latency_p50_s'])

    def test_throughput_uses_wall_time_not_sum_of_user_rates(self):
        samples = [{'ok': True, 'latency_s': 2, 'first_content_s': .5, 'output_tokens': 10}] * 4
        result = summarize(samples, 2)
        self.assertEqual(result['aggregate_tokens_per_s'], 20)
        self.assertEqual(result['latency_p95_s'], 2)

    def test_partial_usage_does_not_produce_inflated_or_partial_rate(self):
        samples = [{'ok': True, 'latency_s': 1, 'first_content_s': None, 'output_tokens': n} for n in [10, None]]
        self.assertIsNone(summarize(samples, 1)['aggregate_tokens_per_s'])

    def test_worker_count_is_bounded_and_failures_counted(self):
        lock = threading.Lock()
        seen = []
        def sample():
            with lock:
                seen.append(1)
            return {'ok': False, 'error': 'synthetic'}
        report = run_level(sample, 2, 7)
        self.assertEqual(len(seen), 7)
        self.assertEqual(report['failed'], 7)


class HTTPTests(unittest.TestCase):
    def test_real_loopback_transport_for_both_protocols(self):
        observed = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                observed.append((self.path, payload))
                self.send_response(200)
                self.end_headers()
                body = (b'{"response":"hello"}\n{"done":true,"eval_count":5}\n'
                        if self.path == '/api/generate' else
                        b'data: {"choices":[{"text":"hello"}]}\n\ndata: {"usage":{"completion_tokens":6}}\n\ndata: [DONE]\n')
                self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            base = 'http://127.0.0.1:' + str(server.server_port)
            for backend, count in [('ollama', 5), ('vllm', 6)]:
                result = request_sample(backend, base, 'fixture', 'synthetic', 12, 2048, 3)
                self.assertTrue(result['ok'])
                self.assertEqual(result['output_tokens'], count)
                self.assertGreaterEqual(result['latency_s'], result['first_content_s'])
            self.assertEqual(observed[0][1]['options']['num_ctx'], 2048)
            self.assertTrue(observed[1][1]['stream_options']['include_usage'])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)


if __name__ == '__main__':
    unittest.main()
