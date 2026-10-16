def anomaly_task_supported(has_temporal_structure: bool, has_evaluation_signal: bool) -> bool:
    return has_temporal_structure and has_evaluation_signal
