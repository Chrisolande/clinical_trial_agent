from agents import supervisor_helpers


def test_unwrap_report_json_merges_nested_report_json() -> None:
    content = '{"report_json": {"x": 1}, "other": 2}'
    parsed = supervisor_helpers.unwrap_report_json(content)
    assert isinstance(parsed, dict)
    assert parsed["report_json"]["x"] == 1


def test_extract_final_result_prefers_report_json() -> None:
    result = {"report_json": {"ok": True}}
    assert supervisor_helpers.extract_final_result(result) == result


def test_extract_final_result_unwraps_last_message_json() -> None:
    class Msg:
        def __init__(self, content: str) -> None:
            self.content = content

    wrapped = {"messages": [Msg('{"report_json": {"a": 1}}')]}
    extracted = supervisor_helpers.extract_final_result(wrapped)
    assert extracted.get("report_json", {}).get("a") == 1


def test_unwrap_synthesis_result_parses_report_text_json_blob() -> None:
    synthesis = {"report_text": '{"report_json": {"a": 2}}'}
    unwrapped = supervisor_helpers.unwrap_synthesis_result(synthesis)
    assert isinstance(unwrapped.get("report_json"), dict)
    assert unwrapped["report_json"]["a"] == 2
