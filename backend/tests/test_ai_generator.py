"""
Tests for the agentic tool loop in AIGenerator and source tracking in the tools.

The Anthropic client is replaced by a fake that returns scripted responses, so
these tests need no API key or network access.
"""

import copy
from types import SimpleNamespace

from ai_generator import AIGenerator, MAX_TOOL_ROUNDS
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults

TOOLS = [{"name": "search_course_content", "input_schema": {"type": "object"}}]


def text_response(text):
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text=text),
        ],
    )


def tool_response(*queries):
    blocks = [
        SimpleNamespace(type="tool_use", id=f"tu_{q}", name="search_course_content", input={"query": q})
        for q in queries
    ]
    return SimpleNamespace(stop_reason="tool_use", content=blocks)


class FakeMessages:
    """Stands in for client.messages: returns scripted responses, records calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **params):
        # Deep-copy: the generator keeps appending to the same messages list.
        self.calls.append(copy.deepcopy(params))
        return self.responses.pop(0)


class FakeToolManager:
    def __init__(self, fail=False):
        self.fail = fail
        self.executed = []

    def execute_tool(self, name, **kwargs):
        self.executed.append(kwargs["query"])
        if self.fail:
            raise RuntimeError("database is down")
        return f"results for {kwargs['query']}"


def make_generator(responses):
    generator = AIGenerator(api_key="test-key", model="test-model")
    generator.client = SimpleNamespace(messages=FakeMessages(responses))
    return generator, generator.client.messages


def tool_results(call):
    """The tool_result blocks sent in the last message of an API call."""
    return call["messages"][-1]["content"]


class TestAgenticLoop:

    def test_no_tool_use_makes_one_call(self):
        generator, messages = make_generator([text_response("Hello")])
        tools = FakeToolManager()
        answer = generator.generate_response("hi", tools=TOOLS, tool_manager=tools)
        assert answer == "Hello"
        assert len(messages.calls) == 1
        assert tools.executed == []

    def test_multiple_tool_rounds_then_answer(self):
        generator, messages = make_generator(
            [tool_response("a"), tool_response("b", "c"), text_response("Done")]
        )
        tools = FakeToolManager()
        answer = generator.generate_response("q", tools=TOOLS, tool_manager=tools)
        assert answer == "Done"
        assert len(messages.calls) == 3
        assert tools.executed == ["a", "b", "c"]
        assert [r["content"] for r in tool_results(messages.calls[1])] == ["results for a"]
        assert [r["tool_use_id"] for r in tool_results(messages.calls[2])] == ["tu_b", "tu_c"]
        # Every exchange is replayed: user, then assistant/user per tool round.
        assert [m["role"] for m in messages.calls[2]["messages"]] == ["user", "assistant", "user", "assistant", "user"]
        assert all(call["tool_choice"] == {"type": "auto"} for call in messages.calls)

    def test_loop_stops_after_max_rounds_and_forces_an_answer(self):
        responses = [tool_response(f"q{i}") for i in range(MAX_TOOL_ROUNDS)] + [text_response("Final")]
        generator, messages = make_generator(responses)
        tools = FakeToolManager()
        answer = generator.generate_response("q", tools=TOOLS, tool_manager=tools)
        assert answer == "Final"
        assert len(tools.executed) == MAX_TOOL_ROUNDS
        assert len(messages.calls) == MAX_TOOL_ROUNDS + 1
        assert all(call["tool_choice"] == {"type": "auto"} for call in messages.calls[:-1])
        assert messages.calls[-1]["tool_choice"] == {"type": "none"}

    def test_system_and_tools_are_identical_on_every_call(self):
        generator, messages = make_generator([tool_response("a"), tool_response("b"), text_response("Done")])
        generator.generate_response("q", conversation_history="earlier chat", tools=TOOLS,
                                    tool_manager=FakeToolManager())
        assert len({call["system"] for call in messages.calls}) == 1
        assert all(call["tools"] == TOOLS for call in messages.calls)

    def test_tool_error_is_reported_and_loop_continues(self):
        generator, messages = make_generator([tool_response("a"), text_response("Sorry")])
        answer = generator.generate_response("q", tools=TOOLS, tool_manager=FakeToolManager(fail=True))
        assert answer == "Sorry"
        [result] = tool_results(messages.calls[1])
        assert result["is_error"] is True
        assert "database is down" in result["content"]

    def test_system_prompt_states_the_search_limit(self):
        assert f"at most {MAX_TOOL_ROUNDS} searches" in AIGenerator.SYSTEM_PROMPT


class FakeStore:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def search(self, query, course_name=None, lesson_number=None):
        return self.results_by_query[query]


def search_results(*metadata):
    return SearchResults(documents=["text"] * len(metadata), metadata=list(metadata),
                         distances=[0.0] * len(metadata))


def test_sources_accumulate_across_searches_until_reset():
    store = FakeStore({
        "mcp": search_results({"course_title": "MCP", "lesson_number": 1}),
        "chroma": search_results({"course_title": "Chroma", "lesson_number": None}),
    })
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(store))
    manager.execute_tool("search_course_content", query="mcp")
    manager.execute_tool("search_course_content", query="chroma")
    assert manager.get_last_sources() == ["MCP - Lesson 1", "Chroma"]
    manager.reset_sources()
    assert manager.get_last_sources() == []
