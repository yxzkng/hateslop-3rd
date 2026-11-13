import openai
import os
import json
import wikipedia
from dotenv import load_dotenv


load_dotenv(override=True)
key = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI()
MODEL = "gpt-4o-mini"

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain import hub

@tool
def add(a: int, b: int) -> int:
    "입력 받은 두 수의 덧셈을 실행합니다."
    return a+b

@tool
def multiply(a: int, b: int) -> int:
    "입력 받은 두 수의 곱셈을 실행합니다."
    return a*b

tools = [add, multiply]
llm = ChatOpenAI(model = MODEL, temperature = 0)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)

print("\n===== LangChain Agent 실행 =====")
question = "5에 7을 더한 값에 3을 곱해줘."
result = agent_executor.invoke({"input": question})

print("\n===== LangChain Agent의 최종 답변 =====")
print(result['output'])


