import sys
import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.config import settings
from core.llm import get_llm

try:
    print("Getting LLM...")
    llm = get_llm(settings.LLM_MODEL, temperature=0.35)
    prompt = ChatPromptTemplate.from_template("Hello")
    chain = prompt | llm | StrOutputParser()
    print("Invoking chain...")
    res = chain.invoke({})
    print("Result:", res)
except Exception as e:
    traceback.print_exc()
