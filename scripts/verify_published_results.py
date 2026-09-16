"""Recompute published benchmark statistics without timing inference."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/evidence/2026-08-25_s4_fp32_benchmark'


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    protocol = json.loads((RESULTS / 'benchmark_protocol.json').read_text(encoding='utf-8'))
    summary = json.loads((RESULTS / 'benchmark_summary.json').read_text(encoding='utf-8'))
    schedule = json.loads((RESULTS / 'run_schedule.json').read_text(encoding='utf-8'))['tasks']
    rows = read_csv(RESULTS / 'raw_latency.csv')
    require(protocol['status'] == 'frozen_before_measurement', 'Protocol status differs')
    require(len(rows) == 4500, 'Expected 4500 measured rows')
    require(len(schedule) == 15, 'Expected 15 workers')
    require(len(list((RESULTS / 'workers').glob('*.csv'))) == 15, 'Worker files incomplete')
    input_hash = hashlib.sha256((RESULTS / 'canonical_input_cat_original.f32').read_bytes()).hexdigest()
    medians = {}
    for task in schedule:
        runtime, repeat, order = task['runtime'], task['repeat_id'], task['run_order']
        subset = [r for r in rows if r['runtime'] == runtime and int(r['repeat_id']) == repeat]
        require(len(subset) == 300, f'{runtime}/{repeat}: expected 300 rows')
        subset.sort(key=lambda r: int(r['iteration']))
        require([int(r['iteration']) for r in subset] == list(range(300)), 'Iteration sequence differs')
        worker = read_csv(RESULTS / 'workers' / f'{order:02}_{runtime}_repeat_{repeat}.csv')
        require(len(worker) == 300, 'Worker row count differs')
        for merged, original in zip(subset, worker):
            require(math.isclose(float(merged['latency_ms']), float(original['latency_ms']), abs_tol=1e-8), 'Merged latency differs')
            require(merged['model_sha256'] == protocol['model']['sha256'], 'Model hash differs')
            require(merged['input_sha256'] == input_hash, 'Input hash differs')
            require((merged['batch_size'], merged['threads'], merged['logical_cpu']) == ('1', '1', '0'), 'Resource configuration differs')
            require(int(merged['run_order']) == order, 'Schedule order differs')
            require(math.isfinite(float(merged['latency_ms'])) and float(merged['latency_ms']) > 0, 'Invalid latency')
        medians.setdefault(runtime, []).append(statistics.median(float(r['latency_ms']) for r in subset))
    primary = {runtime: statistics.median(values) for runtime, values in medians.items()}
    for expected in summary['runtime_summaries']:
        runtime = expected['runtime']
        values = medians[runtime]
        cv = statistics.stdev(values) / statistics.mean(values) * 100
        require(len(values) == 5, 'Repeat count differs')
        require(math.isclose(primary[runtime], expected['primary_latency_median_of_repeat_medians_ms'], abs_tol=1e-7), 'Primary latency differs')
        require(math.isclose(cv, expected['repeat_median_cv_percent'], abs_tol=1e-6), 'CV differs')
        require(cv <= 10, 'Repeat instability')
        require(math.isclose(primary['pytorch'] / primary[runtime], expected['speedup_vs_pytorch_primary'], abs_tol=1e-7), 'Speedup differs')
        print(f'{runtime}: {primary[runtime]:.5f} ms; repeat-median CV {cv:.5f}%')
    print('PASS: 4500 rows, 15 workers, hashes and summary checked; no new timing measurements.')


if __name__ == '__main__':
    main()
