from enacttom.runner.benchmark import BenchmarkRunner


class _Binding:
    def __init__(self, mechanic_type, allowed_targets=None):
        self.mechanic_type = mechanic_type
        self.allowed_targets = allowed_targets


class _Task:
    def __init__(self, num_agents, message_targets=None, mechanic_bindings=None):
        self.num_agents = num_agents
        self.message_targets = message_targets
        self.mechanic_bindings = mechanic_bindings or []


def test_resolve_message_targets_prefers_explicit_field():
    task = _Task(
        num_agents=3,
        message_targets={"0": ["1", "2"]},
        mechanic_bindings=[
            _Binding(
                "restricted_communication",
                {"agent_0": ["agent_1"]},
            )
        ],
    )
    assert BenchmarkRunner._resolve_message_targets(task) == {
        "agent_0": ["agent_1", "agent_2"]
    }


def test_resolve_message_targets_derives_from_restricted_communication():
    task = _Task(
        num_agents=3,
        message_targets=None,
        mechanic_bindings=[
            _Binding(
                "restricted_communication",
                {
                    "agent_0": ["agent_1"],
                    1: [2],  # mixed int/string agent IDs
                    "agent_2": ["agent_2", "agent_0"],  # self target dropped
                },
            )
        ],
    )
    assert BenchmarkRunner._resolve_message_targets(task) == {
        "agent_0": ["agent_1"],
        "agent_1": ["agent_2"],
        "agent_2": ["agent_0"],
    }


def test_resolve_message_targets_none_when_unavailable():
    task = _Task(
        num_agents=2,
        message_targets=None,
        mechanic_bindings=[_Binding("room_restriction", None)],
    )
    assert BenchmarkRunner._resolve_message_targets(task) is None
