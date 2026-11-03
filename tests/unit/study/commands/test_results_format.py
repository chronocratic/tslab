from tslab.study.commands.results import ResultsFormat


def test_results_format_has_the_csv_and_json_forms() -> None:
    assert [member.value for member in ResultsFormat] == ['csv', 'json']
