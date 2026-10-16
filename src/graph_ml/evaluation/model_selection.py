def complexity_justified(
    baseline_score: float,
    candidate_score: float,
    minimum_improvement: float,
) -> bool:
    return (candidate_score - baseline_score) >= minimum_improvement
