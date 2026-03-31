"""Tests for ShortTermMemory."""
import pytest
from lib.memory import ShortTermMemory, SessionNotFoundError


class TestShortTermMemory:

    def test_default_session_created_on_init(self):
        mem = ShortTermMemory()
        assert "default" in mem.get_all_sessions()

    def test_create_session(self):
        mem = ShortTermMemory()
        result = mem.create_session("test-session")
        assert result is True
        assert "test-session" in mem.get_all_sessions()

    def test_create_duplicate_session_returns_false(self):
        mem = ShortTermMemory()
        mem.create_session("dup")
        result = mem.create_session("dup")
        assert result is False

    def test_add_and_get_last_object(self):
        mem = ShortTermMemory()
        mem.add("hello")
        obj = mem.get_last_object()
        assert obj == "hello"

    def test_add_multiple_objects(self):
        mem = ShortTermMemory()
        mem.add("first")
        mem.add("second")
        obj = mem.get_last_object()
        assert obj == "second"

    def test_get_all_objects(self):
        mem = ShortTermMemory()
        mem.add("a")
        mem.add("b")
        mem.add("c")
        objs = mem.get_all_objects()
        assert objs == ["a", "b", "c"]

    def test_get_last_object_empty_session(self):
        mem = ShortTermMemory()
        assert mem.get_last_object() is None

    def test_session_isolation(self):
        mem = ShortTermMemory()
        mem.create_session("session-a")
        mem.create_session("session-b")
        mem.add("item_a", session_id="session-a")
        mem.add("item_b", session_id="session-b")
        assert mem.get_last_object("session-a") == "item_a"
        assert mem.get_last_object("session-b") == "item_b"

    def test_add_to_nonexistent_session_raises(self):
        mem = ShortTermMemory()
        with pytest.raises(SessionNotFoundError):
            mem.add("something", session_id="ghost-session")

    def test_reset_clears_session(self):
        mem = ShortTermMemory()
        mem.add("value")
        mem.reset()
        assert mem.get_last_object() is None

    def test_delete_session(self):
        mem = ShortTermMemory()
        mem.create_session("to-delete")
        mem.delete_session("to-delete")
        assert "to-delete" not in mem.get_all_sessions()

    def test_cannot_delete_default_session(self):
        mem = ShortTermMemory()
        with pytest.raises(ValueError):
            mem.delete_session("default")

    def test_pop_removes_last_item(self):
        mem = ShortTermMemory()
        mem.add("first")
        mem.add("second")
        popped = mem.pop()
        assert popped == "second"
        assert mem.get_last_object() == "first"

    def test_pop_empty_session_returns_none(self):
        mem = ShortTermMemory()
        assert mem.pop() is None

    def test_objects_are_deep_copied(self):
        mem = ShortTermMemory()
        data = {"key": "value"}
        mem.add(data)
        data["key"] = "modified"
        stored = mem.get_last_object()
        assert stored["key"] == "value"
