import re

PATTERNS = {
    "Security Alert": [
        r"login.{0,20}fail",
        r"unauthorized.{0,20}access",
        r"brute.?force",
        r"account.{0,20}locked",
        r"ssl.{0,20}certif",
        r"permission.{0,20}denied",
        r"security.{0,20}breach",
        r"attack.{0,20}detect",
    ],
    "Resource Usage": [
        r"phys_ram=\d+",
        r"used_ram=\d+",
        r"\[instance:.*\]\s+total memory",
        r"cpu.{0,20}utiliz",
        r"disk.{0,20}usage.{0,20}\d+%",
        r"memory.{0,20}pressure",
        r"connection.{0,20}pool.{0,20}exhaust",
        r"network.{0,20}latency",
    ],
    "Workflow Error": [
        r"escalation.{0,20}fail",
        r"task.{0,20}assign.{0,20}could not",
        r"workflow.{0,20}(step|stage).{0,20}(fail|timeout|timed out)",
        r"pipeline.{0,20}(fail|error)",
        r"job.{0,20}scheduler.{0,20}fail",
        r"health.{0,20}check.{0,20}fail",
        r"undefined escalation",
        r"invalid priority",
    ],
}


class RegexClassifier:
    def predict(self, text: str) -> str | None:
        lower = text.lower()
        for label, patterns in PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, lower):
                    return label
        return None
