from dataclasses import dataclass

@dataclass(frozen=True)
class ComputeBenchmark:
    model_name: str
    train_seconds: float
    inference_ms_per_record: float
    parameter_count: int
