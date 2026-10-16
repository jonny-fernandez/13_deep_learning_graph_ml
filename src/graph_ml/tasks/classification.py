def validate_binary_target(values: list[int]) -> None:
    invalid = sorted(set(values) - {0, 1})
    if invalid:
        raise ValueError(f"Binary target contains invalid values: {invalid}")
