# Solely for test purpose

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    api_key="sk-xxxx",
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-20b:free",
    # max_tokens=4096,
    max_retries=3,
    timeout=60,
)
# Schema for structured output
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    answer: int | None = Field(None, description="Answer to the user's request.")
    justification: str | None = Field(None, description="Why the answer is what it is.")
    thinking_process: str | None = Field(
        None, description="The reasoning steps taken to arrive at the answer."
    )
# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
output = structured_llm.invoke("what is 29*31 equal to?")
print(f'output: {output}')

# Define a tool
def multiply(a: int, b: int) -> int:
    return a * b


# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("What is 22222 times 391827?")
print(f"msg: {msg.content}")
