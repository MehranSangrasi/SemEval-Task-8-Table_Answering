from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()


ChatOpenAI.openai_api_key=os.getenv("OPENAI_API_KEY")
df = pd.read_csv("train (1).csv")
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent_executor = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type="tool-calling",
    verbose=True,
    allow_dangerous_code=True
)

result = agent_executor.run("What are the top 4 fairs paid by passengers?")
print(result)