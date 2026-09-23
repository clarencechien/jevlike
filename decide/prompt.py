"""Render state + typed question into a prompt that stops at the start of the assistant turn.

The chat template is learned from the running llama-server (/apply-template) once, using
marker strings, so per-call rendering is a local string substitution with no server round-trip.
"""
import requests

SYSTEM = "你是產線決策引擎。只回答一個字母。"
LETTERS = "ABCDEFGHIJ"
SYS_MARK, USR_MARK = "§SYSMARK§", "§USRMARK§"


def render_question(question: dict) -> str:
    """question = {instructions, options: {A: label | {label, criteria}}}"""
    lines = [f"【問題】{question['instructions']}"]
    for letter, opt in question["options"].items():
        if isinstance(opt, dict):
            crit = opt.get("criteria")
            lines.append(f"{letter}. {opt['label']}" + (f"：{crit}" if crit else ""))
        else:
            lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


def build_messages(state: str, question: dict, system: str = SYSTEM) -> list[dict]:
    user = f"【狀態】\n{state}\n\n{render_question(question)}\n\n只回答代表正確選項的字母。"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class TemplateRenderer:
    def __init__(self, base_url: str, prefill: str = "答案：", use_system: bool = True):
        self.base_url = base_url
        self.prefill = prefill
        self.use_system = use_system
        msgs = [{"role": "system", "content": SYS_MARK}, {"role": "user", "content": USR_MARK}]
        if not use_system:
            msgs = msgs[1:]
        r = requests.post(f"{base_url}/apply-template", json={"messages": msgs}, timeout=60)
        r.raise_for_status()
        self.template = r.json()["prompt"]
        if USR_MARK not in self.template or (use_system and SYS_MARK not in self.template):
            raise RuntimeError(f"apply-template lost markers: {self.template!r}")

    def render(self, state: str, question: dict, system: str = SYSTEM, prefill: str | None = None) -> str:
        msgs = build_messages(state, question, system)
        out = self.template.replace(USR_MARK, msgs[1]["content"])
        if self.use_system:
            out = out.replace(SYS_MARK, msgs[0]["content"])
        return out + (self.prefill if prefill is None else prefill)

    def render_messages(self, msgs: list[dict], prefill: str | None = None) -> str:
        """Generic: msgs = [system?, user]."""
        sys_c = next((m["content"] for m in msgs if m["role"] == "system"), "")
        usr_c = next(m["content"] for m in msgs if m["role"] == "user")
        out = self.template.replace(USR_MARK, usr_c)
        if self.use_system:
            out = out.replace(SYS_MARK, sys_c)
        return out + (self.prefill if prefill is None else prefill)


def letters_for(n: int) -> list[str]:
    return list(LETTERS[:n])
