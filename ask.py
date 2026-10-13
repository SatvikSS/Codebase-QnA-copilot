import sys
import time

from src.agent.graph import run_agent
from src.rag.answer import answer_question

args = sys.argv[1:]
use_agent = "--agent" in args
question = " ".join(arg for arg in args if arg != "--agent")

start = time.perf_counter()
if use_agent:
    result = run_agent(question)
    answer, responses = result.answer, result.responses
    print("Agent steps:")
    for step in result.trace:
        print(f"  {step}")
    print()
else:
    answer = answer_question(question)
    responses = (answer.response,)
elapsed = time.perf_counter() - start

print(answer.text)

print("\nCitations:")
for citation in answer.citations or ("(none)",):
    print(f"  {citation}")

tokens_in = sum(r.prompt_tokens for r in responses)
tokens_out = sum(r.completion_tokens for r in responses)
cost = sum(r.cost or 0 for r in responses)
print(f"\n{elapsed:.1f}s | {len(responses)} LLM calls | {tokens_in} in / {tokens_out} out tokens | cost ${cost:.5f}")
