import sys
from core.llm import get_llm
try:
    llm = get_llm("gemini-flash-latest", temperature=0.35)
    print(llm.invoke("Hello"))
except Exception as e:
    with open("error.log", "w") as f:
        f.write(str(e))
