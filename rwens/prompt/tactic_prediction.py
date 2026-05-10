from rwens.prompt.base import PromptFormatter


class BaseStateToTacPromptFormatter(PromptFormatter):

    BASE_STATE_TO_TAC_PROMPT = """
You are a helpful assistant that predicts the next Lean4 tactic to be applied in a proof.
You are given a state of a proof and are tasked with predicting the next tactic to be applied.

Here is the state of the proof:
{state}

Answer in the following format:
```lean4
tactic_code
```
"""

    def format(self, state: str) -> str:
        return self.BASE_STATE_TO_TAC_PROMPT.format(state=state)

    def format_answer(self, answer: str) -> str:
        return f"```lean4\n{answer}\n```"

    def extract_answer(self, response: str) -> str:
        parts = response.split("```lean4")
        if len(parts) < 2:
            return response.strip()
        return parts[1].split("```")[0].strip()
