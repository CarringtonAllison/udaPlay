"""Tests for the StateMachine framework."""
import pytest
from typing import TypedDict, Optional
from lib.state_machine import (
    StateMachine, Step, EntryPoint, Termination, Run,
    Snapshot, Resource, Transition
)


class SimpleState(TypedDict):
    value: int
    message: str


def increment_step(state: SimpleState) -> SimpleState:
    return {"value": state["value"] + 1}


def double_step(state: SimpleState) -> SimpleState:
    return {"value": state["value"] * 2}


def _build_simple_machine():
    machine = StateMachine[SimpleState](SimpleState)
    entry = EntryPoint[SimpleState]()
    step1 = Step[SimpleState]("increment", increment_step)
    termination = Termination[SimpleState]()
    machine.add_steps([entry, step1, termination])
    machine.connect(entry, step1)
    machine.connect(step1, termination)
    return machine


class TestStep:

    def test_step_str(self):
        s = Step("my_step", lambda x: {})
        assert "my_step" in str(s)

    def test_step_run_1_arg(self):
        s = Step[SimpleState]("inc", increment_step)
        state: SimpleState = {"value": 5, "message": ""}
        result = s.run(state, SimpleState)
        assert result["value"] == 6

    def test_step_run_2_arg(self):
        def with_resource(state, resource):
            return {"value": resource.vars["multiplier"] * state["value"]}

        s = Step[SimpleState]("mul", with_resource)
        resource = Resource(vars={"multiplier": 3})
        state: SimpleState = {"value": 4, "message": ""}
        result = s.run(state, SimpleState, resource)
        assert result["value"] == 12

    def test_entry_point_is_special(self):
        entry = EntryPoint[SimpleState]()
        assert entry.step_id == "__entry__"

    def test_termination_is_special(self):
        term = Termination[SimpleState]()
        assert term.step_id == "__termination__"


class TestRun:

    def test_run_create(self):
        run = Run.create()
        assert run.run_id is not None
        assert run.start_timestamp is not None
        assert run.end_timestamp is None

    def test_run_complete(self):
        run = Run.create()
        run.complete()
        assert run.end_timestamp is not None

    def test_run_get_final_state_empty(self):
        run = Run.create()
        assert run.get_final_state() is None

    def test_run_add_snapshot_and_get_final(self):
        run = Run.create()
        snap = Snapshot.create({"value": 42, "message": ""}, SimpleState, "test_step")
        run.add_snapshot(snap)
        assert run.get_final_state()["value"] == 42

    def test_run_metadata(self):
        run = Run.create()
        run.complete()
        meta = run.metadata
        assert "run_id" in meta
        assert "snapshot_counts" in meta


class TestStateMachine:

    def test_simple_machine_runs(self):
        machine = _build_simple_machine()
        initial: SimpleState = {"value": 10, "message": ""}
        run = machine.run(initial)
        assert run.get_final_state()["value"] == 11

    def test_run_returns_run_object(self):
        machine = _build_simple_machine()
        initial: SimpleState = {"value": 1, "message": ""}
        result = machine.run(initial)
        assert isinstance(result, Run)

    def test_snapshots_recorded(self):
        machine = _build_simple_machine()
        initial: SimpleState = {"value": 1, "message": ""}
        run = machine.run(initial)
        # entry + increment step = at least 2 snapshots
        assert len(run.snapshots) >= 1

    def test_no_entry_point_raises(self):
        machine = StateMachine[SimpleState](SimpleState)
        s = Step[SimpleState]("only", lambda x: {})
        machine.add_steps([s])
        with pytest.raises(Exception, match="No EntryPoint"):
            machine.run({"value": 1, "message": ""})

    def test_conditional_transition(self):
        """Test that conditional transitions route correctly."""
        class BranchState(TypedDict):
            flag: bool
            result: str

        machine = StateMachine[BranchState](BranchState)
        entry = EntryPoint[BranchState]()
        yes_step = Step[BranchState]("yes", lambda s: {"result": "yes"})
        no_step = Step[BranchState]("no", lambda s: {"result": "no"})
        term = Termination[BranchState]()

        machine.add_steps([entry, yes_step, no_step, term])
        machine.connect(entry, [yes_step, no_step],
                        lambda s: yes_step if s["flag"] else no_step)
        machine.connect(yes_step, term)
        machine.connect(no_step, term)

        run_yes = machine.run({"flag": True, "result": ""})
        assert run_yes.get_final_state()["result"] == "yes"

        run_no = machine.run({"flag": False, "result": ""})
        assert run_no.get_final_state()["result"] == "no"

    def test_resource_passed_to_steps(self):
        class ResState(TypedDict):
            output: str

        def use_resource(state, resource):
            return {"output": resource.vars["greeting"]}

        machine = StateMachine[ResState](ResState)
        entry = EntryPoint[ResState]()
        s = Step[ResState]("greet", use_resource)
        term = Termination[ResState]()
        machine.add_steps([entry, s, term])
        machine.connect(entry, s)
        machine.connect(s, term)

        resource = Resource(vars={"greeting": "hello world"})
        run = machine.run({"output": ""}, resource=resource)
        assert run.get_final_state()["output"] == "hello world"
