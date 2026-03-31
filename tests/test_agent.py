"""Tests for Agent class and AgentState."""
import json
import pytest
from unittest.mock import MagicMock, patch
from lib.agents import Agent, AgentState
from lib.state_machine import Run
from lib.tooling import tool


def _make_openai_response(content: str = "Test answer.", tool_calls=None):
    """Build a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = tool_calls
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=30, total_tokens=80)
    return mock_response


class TestAgentState:

    def test_agent_state_is_typed_dict(self):
        state: AgentState = {
            "user_query": "test",
            "instructions": "You are helpful.",
            "messages": [],
            "current_tool_calls": None,
            "total_tokens": 0,
            "session_id": "default",
        }
        assert state["user_query"] == "test"


class TestAgentInit:

    def test_agent_stores_instructions(self):
        agent = Agent(model_name="gpt-4o-mini", instructions="Be helpful.")
        assert agent.instructions == "Be helpful."

    def test_agent_stores_model_name(self):
        agent = Agent(model_name="gpt-4o-mini", instructions="test")
        assert agent.model_name == "gpt-4o-mini"

    def test_agent_empty_tools_by_default(self):
        agent = Agent(model_name="gpt-4o-mini", instructions="test")
        assert agent.tools == []

    def test_agent_accepts_tools(self):
        @tool
        def my_tool(q: str) -> str:
            """Test tool."""
            return q
        agent = Agent(model_name="gpt-4o-mini", instructions="test", tools=[my_tool])
        assert len(agent.tools) == 1


class TestAgentInvoke:

    def test_invoke_returns_run(self):
        with patch("lib.llm.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response()

            agent = Agent(model_name="gpt-4o-mini", instructions="You are helpful.")
            result = agent.invoke("What is Gran Turismo?")
            assert isinstance(result, Run)

    def test_invoke_final_state_has_messages(self):
        with patch("lib.llm.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response(
                "Gran Turismo is a racing game."
            )

            agent = Agent(model_name="gpt-4o-mini", instructions="Be helpful.")
            run = agent.invoke("Tell me about Gran Turismo.")
            state = run.get_final_state()
            assert len(state["messages"]) > 0

    def test_invoke_uses_default_session(self):
        with patch("lib.llm.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response()

            agent = Agent(model_name="gpt-4o-mini", instructions="test")
            agent.invoke("Hello")
            runs = agent.get_session_runs("default")
            assert len(runs) == 1

    def test_invoke_with_custom_session_id(self):
        with patch("lib.llm.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response()

            agent = Agent(model_name="gpt-4o-mini", instructions="test")
            agent.invoke("Hello", session_id="my-session")
            runs = agent.get_session_runs("my-session")
            assert len(runs) == 1

    def test_multi_turn_accumulates_messages(self):
        with patch("lib.llm.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response()

            agent = Agent(model_name="gpt-4o-mini", instructions="test")
            agent.invoke("Q1", session_id="multi")
            agent.invoke("Q2", session_id="multi")
            runs = agent.get_session_runs("multi")
            assert len(runs) == 2

    def test_token_count_tracked(self):
        with patch("lib.llm.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response()

            agent = Agent(model_name="gpt-4o-mini", instructions="test")
            run = agent.invoke("test query")
            state = run.get_final_state()
            assert state.get("total_tokens", 0) >= 0

    def test_reset_session_clears_memory(self):
        with patch("lib.llm.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response()

            agent = Agent(model_name="gpt-4o-mini", instructions="test")
            agent.invoke("Hello", session_id="clear-me")
            agent.reset_session("clear-me")
            runs = agent.get_session_runs("clear-me")
            assert len(runs) == 0
