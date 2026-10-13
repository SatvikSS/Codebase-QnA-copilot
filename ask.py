import sys
import time

from src.rag.answer import answer_question

question = " ".join(sys.argv[1:])

start = time.perf_counter()
result = answer_question(question)
elapsed = time.perf_counter() - start

print(result.text)

print("\nCitations:")
for citation in result.citations or ("(none)",):
    print(f"  {citation}")

usage = result.response
cost = f"${usage.cost:.5f}" if usage.cost is not None else "n/a"
print(f"\n{elapsed:.1f}s | {usage.prompt_tokens} in / {usage.completion_tokens} out tokens | cost {cost}")
