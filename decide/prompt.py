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


ANSWER_LINE = {"letter": "只回答代表正確選項的字母。",
               "json": '以 {"answer": "<字母>"} 回答，只輸出這個 JSON。'}
JSON_PREFILL = '{"answer": "'  # T1 (handoff v5): TypeLLM-style label prefill; the next token is the letter

# T1 variants: V0 current, V1 prefill only, V2 prefill + instruction
VARIANTS = {"V0": {"prefill": "", "answer_style": "letter"},
            "V1": {"prefill": JSON_PREFILL, "answer_style": "letter"},
            "V2": {"prefill": JSON_PREFILL, "answer_style": "json"}}


def build_messages(state: str, question: dict, system: str = SYSTEM, answer_style: str = "letter") -> list[dict]:
    user = f"【狀態】\n{state}\n\n{render_question(question)}\n\n{ANSWER_LINE[answer_style]}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


GEMMA4_TEMPLATE_NOTHINK = ("<|turn>system\n" + SYS_MARK + "<turn|>\n<|turn>user\n" + USR_MARK + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")


class TemplateRenderer:
    """Learns the server's chat template once (with thinking disabled) and renders locally.

    Gemma 4 template: enable_thinking=True injects '<|think|>\n' at the top of the system turn;
    enable_thinking=False omits it and appends an empty thought block
    '<|channel>thought\n<channel|>' after '<|turn>model\n' so the next token is the answer.
    """

    THINK_ON = "<|think|>\n"
    EMPTY_THOUGHT = "<|channel>thought\n<channel|>"

    def __init__(self, base_url: str, prefill: str = "", use_system: bool = True, enable_thinking: bool = False, static: bool = False,
                 answer_style: str = "letter"):
        self.base_url = base_url
        self.prefill = prefill
        self.answer_style = answer_style
        self.use_system = use_system
        self.enable_thinking = enable_thinking
        if static:  # backend has no /apply-template (e.g. SGLang): use the template llama-server rendered for Gemma 4
            self.template = GEMMA4_TEMPLATE_NOTHINK
            self.server_honored_kwargs = True
            return
        msgs = [{"role": "system", "content": SYS_MARK}, {"role": "user", "content": USR_MARK}]
        if not use_system:
            msgs = msgs[1:]
        r = requests.post(f"{base_url}/apply-template",
                          json={"messages": msgs, "chat_template_kwargs": {"enable_thinking": enable_thinking}}, timeout=60)
        r.raise_for_status()
        t = r.json()["prompt"]
        self.server_honored_kwargs = True
        if not enable_thinking and self.THINK_ON in t:
            # server ignored chat_template_kwargs; patch by hand to match the template's own logic
            self.server_honored_kwargs = False
            t = t.replace(self.THINK_ON, "", 1)
            if not t.endswith(self.EMPTY_THOUGHT):
                t = t + self.EMPTY_THOUGHT
        self.template = t
        if USR_MARK not in self.template or (use_system and SYS_MARK not in self.template):
            raise RuntimeError(f"apply-template lost markers: {self.template!r}")

    def render(self, state: str, question: dict, system: str = SYSTEM, prefill: str | None = None) -> str:
        msgs = build_messages(state, question, system, self.answer_style)
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
