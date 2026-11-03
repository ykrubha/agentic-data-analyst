import json

import pytest

from analyst_agent.agent import AgentConfig, DataAnalystAgent, recover_final_answer
from fakes import ScriptedProvider, final

DB = "data/chinook.sqlite"


def run(turns, config=None, tmp=None):
    provider = ScriptedProvider(turns)
    agent = DataAnalystAgent(provider, DB, config=config, output_dir=tmp)
    return agent.run("q"), provider


def test_happy_path_sql_then_answer():
    result, _ = run([
        [("run_sql", {"query": "SELECT COUNT(*) n FROM Customer", "save_as": "c"})],
        final(59),
    ])
    assert result.status == "answered"
    assert result.value == 59
    assert result.steps == 2
    assert result.sql == ["SELECT COUNT(*) n FROM Customer"]


def test_progress_is_reported_as_steps_happen():
    lines = []
    provider = ScriptedProvider([[("list_tables", {})], final(1)])
    DataAnalystAgent(provider, DB).run("q", on_event=lines.append)
    assert lines[0] == "[1] waiting for the model..."
    assert any("model replied" in line for line in lines)
    assert any(line.startswith("[1] list_tables") for line in lines)
    assert lines[-1].startswith("[2] final_answer")


def test_sql_error_goes_back_to_model_and_recovers():
    result, provider = run([
        [("run_sql", {"query": "SELECT nope FROM Customer", "save_as": "c"})],
        [("run_sql", {"query": "SELECT COUNT(*) FROM Customer", "save_as": "c"})],
        final(59),
    ])
    assert result.tool_errors == 1
    assert result.recovered_errors == 1
    assert result.status == "answered"
    tool_message = next(m for m in provider.calls[1] if m["role"] == "tool")
    assert tool_message["is_error"] is True
    assert "nope" in tool_message["content"]


def test_dataframe_flows_from_sql_to_python():
    code = "print(int(c['Total'].sum() > 0), len(c))"
    result, provider = run([
        [("run_sql", {"query": "SELECT Total FROM Invoice", "save_as": "c"})],
        [("run_python", {"code": code})],
        final(1),
    ])
    assert result.tool_errors == 0
    outcome = json.loads([m for m in provider.calls[2] if m["role"] == "tool"][-1]["content"])
    assert outcome["stdout"].split() == ["1", "412"]


def test_chart_is_saved(tmp_path):
    code = "import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.savefig('trend.png')"
    result, _ = run([[("run_python", {"code": code})], final(None)], tmp=tmp_path)
    assert len(result.charts) == 1
    assert (tmp_path / result.charts[0].split("/")[-1]).exists()


def test_write_attempt_is_refused_with_error():
    result, _ = run([
        [("run_sql", {"query": "DELETE FROM Invoice", "save_as": "x"})],
        final(None, answerable=False),
    ])
    assert result.tool_errors == 1
    assert result.answerable is False


def test_stops_at_step_budget():
    loop = [[("list_tables", {})]] * 50
    result, _ = run(loop, config=AgentConfig(max_steps=4))
    assert result.status == "max_steps"
    assert result.steps == 4


def test_stops_after_too_many_errors():
    bad = [[("run_sql", {"query": "SELECT nope", "save_as": "x"})]] * 50
    result, _ = run(bad, config=AgentConfig(max_tool_errors=3))
    assert result.status == "too_many_errors"
    assert result.tool_errors == 3


def test_stops_at_token_budget():
    loop = [[("list_tables", {})]] * 50
    result, _ = run(loop, config=AgentConfig(max_total_tokens=20))
    assert result.status == "token_budget"


def test_plain_text_turn_is_nudged_not_treated_as_answer():
    result, provider = run(["I think it is 59.", final(59)])
    assert result.status == "answered"
    assert provider.calls[1][-1]["role"] == "user"


def test_text_reply_narrows_tools_to_final_answer_then_restores_them():
    result, provider = run([
        "The answer is Rock.",
        final("Rock"),
    ])
    assert result.status == "answered"
    assert "run_sql" in provider.tools_offered[0]
    assert provider.tools_offered[1] == ["final_answer"]

    _, provider = run(["thinking", [("list_tables", {})], final(1)])
    assert provider.tools_offered[2] != ["final_answer"]


JSON_ANSWER = '{"answer": "Rock", "answerable": true, "value": "Rock"}'


def test_final_answer_written_as_json_text_is_recovered_and_flagged():
    result, provider = run([f"Reasoning here.\n</think>\n\n{JSON_ANSWER}"])
    assert result.status == "answered"
    assert result.value == "Rock"
    assert result.answer_recovered_from_text is True
    assert len(provider.calls) == 1


def test_native_final_answer_is_not_flagged_as_recovered():
    result, _ = run([final(1)])
    assert result.answer_recovered_from_text is False


@pytest.mark.parametrize("text,value", [
    (JSON_ANSWER, "Rock"),
    (f"Here you go: {JSON_ANSWER} Done.", "Rock"),
    (f'{{"name": "final_answer", "arguments": {JSON_ANSWER}}}', "Rock"),
    ('{"answer": "a", "answerable": true, "value": 1} {"answer": "b", "answerable": true, "value": 2}', 2),
    ('{"answer": "x {y}", "answerable": false, "value": null}', None),
])
def test_recover_final_answer_accepts(text, value):
    assert recover_final_answer(text)["value"] == value


@pytest.mark.parametrize("text", [
    "The answer is Rock.",
    "",
    "{not json",
    '{"answer": "Rock"}',
    '{"answer": "Rock", "answerable": "yes"}',
    '{"answer": 5, "answerable": true}',
    '{"name": "run_sql", "arguments": {"query": "SELECT 1"}}',
    # A draft inside the reasoning isn't an answer.
    f"Maybe {JSON_ANSWER} would work.</think>\nStill checking.",
])
def test_recover_final_answer_rejects(text):
    assert recover_final_answer(text) is None


def test_repeated_text_replies_stop_early_and_keep_the_text():
    result, provider = run(["It is Rock."] * 10, config=AgentConfig(max_text_turns=3))
    assert result.status == "no_final_answer"
    assert result.answer == "It is Rock."
    assert result.value is None
    assert len(provider.calls) == 3


def test_unknown_tool_and_bad_args_are_reported():
    result, _ = run([
        [("hack_the_planet", {})],
        [("run_sql", {"wrong_argument": 1})],
        final(None, answerable=False),
    ])
    assert result.tool_errors == 2
    assert result.status == "answered"


def test_incomplete_final_answer_is_rejected_then_corrected():
    result, _ = run([[("final_answer", {"value": 1})], final(1)])
    assert result.status == "answered"
    assert result.tool_errors == 1
    assert result.steps == 2


def test_parallel_tool_calls_all_get_results():
    result, provider = run([
        [("list_tables", {}), ("describe_table", {"table": "Genre"})],
        final(1),
    ])
    assert sum(1 for m in provider.calls[1] if m["role"] == "tool") == 2
    assert result.status == "answered"


@pytest.mark.parametrize("save_as", ["1bad", "a b", "__import__('os')", ""])
def test_save_as_must_be_identifier(save_as):
    result, _ = run([
        [("run_sql", {"query": "SELECT 1", "save_as": save_as})],
        final(1),
    ])
    assert result.tool_errors == 1
