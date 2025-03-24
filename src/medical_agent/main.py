import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
from openai import OpenAI  # Update import
from agent import build_medical_agent, AgentState  # Add this import

# Load environment variables
load_dotenv()

# Define the system prompt
SYSTEM_PROMPT = "你是一个医疗助手，你的主要任务是帮助医生整理病例，诊断报告等文件或图片，并将其整理成结构化数据存储起来。必要时，你也可以回答用户的医疗问题。如果可能，在回答医疗问题时请尽量提供来源。"


def main():
    # Build the agent
    medical_agent = build_medical_agent()
    
    # Use text input
    initial_state = AgentState()
    
    # Run the agent
    result = medical_agent.invoke(initial_state)
    
    # Print the result
    # print(result)

if __name__ == "__main__":
    main()