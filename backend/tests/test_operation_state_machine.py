"""State-machine coverage for admission fences across arbitrary action sequences."""

import time

from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from backend.operations.action_recovery import admit_action, state_token


@settings(max_examples=40, stateful_step_count=30, deadline=None)
class ActionAdmissionMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.sequence = 0

    @rule()
    def record_unknown_attempt(self):
        if not admit_action(self.rows, "engine:test")["admit"]:
            return
        self.sequence += 1
        self.rows.append(
            {
                "operation_id": f"unknown-{self.sequence}",
                "kind": "build",
                "status": "unknown",
                "resource_key": "engine:test",
                "updated_at": time.time() + self.sequence,
                "detail": {"effect_started": True},
            }
        )

    @rule()
    def record_proven_not_started_attempt(self):
        if not admit_action(self.rows, "engine:test")["admit"]:
            return
        self.sequence += 1
        self.rows.append(
            {
                "operation_id": f"negative-{self.sequence}",
                "kind": "build",
                "status": "interrupted",
                "resource_key": "engine:test",
                "updated_at": time.time() + self.sequence,
                "detail": {"effect_started": False},
            }
        )

    @rule()
    def confirm_latest_unresolved_and_finish_replacement(self):
        denied = admit_action(self.rows, "engine:test")
        if denied["code"] != "ACTION_RETRY_WITHHELD":
            return
        operation_id = denied["operation_id"]
        token = denied["state_token"]
        confirmed = admit_action(
            self.rows,
            "engine:test",
            confirm_operation_id=operation_id,
            confirm_state=token,
        )
        assert confirmed["admit"] is True
        old = next(row for row in self.rows if row["operation_id"] == operation_id)
        old.setdefault("detail", {})["confirmation_consumed"] = True
        self.sequence += 1
        replacement = {
            "operation_id": f"replacement-{self.sequence}",
            "kind": "build",
            "status": "running",
            "resource_key": "engine:test",
            "updated_at": time.time() + self.sequence,
            "detail": {"effect_started": False},
        }
        self.rows.append(replacement)
        assert admit_action(self.rows, "engine:test")["code"] == "ACTION_IN_FLIGHT"
        replacement["status"] = "succeeded"

    @invariant()
    def every_unconsumed_unknown_row_keeps_the_gate_closed(self):
        unresolved = [
            row
            for row in self.rows
            if row["status"] == "unknown" and not row.get("detail", {}).get("confirmation_consumed")
        ]
        if unresolved:
            assert admit_action(self.rows, "engine:test")["admit"] is False


TestActionAdmissionStateMachine = ActionAdmissionMachine.TestCase
