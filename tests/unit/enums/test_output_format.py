from tslab.enums import OutputFormat


def test_output_format_has_the_text_and_json_forms() -> None:
    assert [member.value for member in OutputFormat] == ['text', 'json']
