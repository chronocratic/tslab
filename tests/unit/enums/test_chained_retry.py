from tslab import enums


def test_chained_retry_parses_profile_values() -> None:
    assert 'ChainedRetry' in enums.__all__
    assert enums.ChainedRetry('manual') is enums.ChainedRetry.MANUAL
    assert enums.ChainedRetry('resubmit') is enums.ChainedRetry.RESUBMIT
    assert enums.ChainedRetry('requeue') is enums.ChainedRetry.REQUEUE
