from tslab import enums


def test_retry_parses_profile_values() -> None:
    assert 'Retry' in enums.__all__
    assert enums.Retry('manual') is enums.Retry.MANUAL
    assert enums.Retry('resubmit') is enums.Retry.RESUBMIT
