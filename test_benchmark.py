from src.benchmark.benchmark import run_benchmark

results = run_benchmark("data/sample_docs/structured_arabic.txt")

for key, value in results.items():
    print(f"{key}: {value}")