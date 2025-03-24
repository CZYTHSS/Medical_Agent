from typing import List, Dict, Any, TypedDict, Literal, Union
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
import base64
from pathlib import Path
from openai import OpenAI
import os
from medical_agent.utils import call_qwen_vl_api

# Define message types
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: Union[str, Dict[str, Any]]
    content_type: Literal["text", "image"] = "text"

# Define the state
class AgentState(TypedDict):
    messages: List[Message]
    qwen: Any  # For Qwen model
    gpt: Any   # For GPT model
    medical_context: Dict[str, Any]
    reasoning: str
    next: str

# Define the system prompt
SYSTEM_PROMPT = """你是一个医疗助手，你的主要任务是帮助医生整理病例，诊断报告等文件或图片，并将其整理成结构化数据存储起来。必要时，你也可以回答用户的医疗问题。如果可能，在回答医疗问题时请尽量提供来源。"""

from typing import TypedDict, get_type_hints, Any

class AgentState(TypedDict):
    messages: list
    qwen: Any
    gpt: Any
    medical_context: dict
    reasoning: str
    next: str

# Dynamically create an initial state with sensible defaults
def init_typed_dict(cls: TypedDict):
    hints = get_type_hints(cls)
    default_values = {
        list: [],
        dict: {},
        str: "",
        int: 0,
        float: 0.0,
        bool: False,
    }

    state = {}
    for key, hint in hints.items():
        # handle special case of typing.List, typing.Dict, etc.
        origin = getattr(hint, '__origin__', hint)
        state[key] = default_values.get(origin, None)
    return state


def init_llms(state: AgentState):
    """
    Initialize Qwen and OpenAI models in the state
    """
    state = init_typed_dict(AgentState)

    # Initialize Qwen model. Just use the OpenAI API for now. Because there is no guarantee langchain supports qwen well
    state["qwen"] = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # Initialize GPT model
    state["gpt"] = ChatOpenAI(
        model="gpt-4o",
        max_tokens=1024,
        temperature=0.7
    )

    state['messages'].append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })

    
    return state

# Define the response generation node
def create_input_node(state: AgentState):
    """Create a node for handling user input and generating responses."""
    
    # Default values
    default_image_path = "/Users/czy/Desktop/sidegig/Medical_Agent/data/input_1.jpg"
    default_question = "请分析这张医疗图像并提供诊断建议。"
    
    # Get user input or use defaults
    try:
        image_path = input("请输入图片路径 (直接回车使用默认路径): ").strip() or default_image_path
        question = input("请输入您的问题 (直接回车使用默认问题): ").strip() or default_question
    except:
        # If any error occurs during input, use defaults
        image_path = default_image_path
        question = default_question
    
    # Read and encode the image
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"警告：找不到图片 {image_path}，使用默认图片")
            image_path = Path(default_image_path)
            
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Create image content in the format expected by the API
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        
        # Add the message to state
        state["messages"].append({
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": question}
            ],
            "content_type": "image"
        })
        
    except Exception as e:
        print(f"处理图片时出错: {e}")
        # If image processing fails, fall back to text-only
        state["messages"].append({
            "role": "user",
            "content": question,
            "content_type": "text"
        })
    
    return state

# Define the response generation node
def create_response_node(state: AgentState):
    """Create a node for handling user input and generating responses."""
    # Get the last user message
    last_message = next((m for m in reversed(state["messages"]) if m["role"] == "user"), None)
    
    if not last_message:
        return state
        
    # Get content and content type
    content = last_message["content"]
    content_type = last_message.get("content_type", "text")
    
    # Create system message
    system_message = {
        "role": "system", 
        "content": "You are a knowledgeable medical assistant. Your goal is to provide accurate, helpful information about medical topics. Always clarify that you're not a doctor and your advice doesn't replace professional medical consultation."
    }
    
    # Create user message based on content type
    if content_type == "image":
        user_message = {
            "role": "user",
            "content": content
        }
    else:
        user_message = {
            "role": "user",
            "content": content
        }
    
    # Get response from LLM using state's llm
    # result = state["gpt"].invoke([system_message, user_message])
    completion = call_qwen_vl_api(state)
    
    # Add the response to messages
    state["messages"].append({
        "role": "assistant",
        "content": completion,
        "content_type": "text"
    })
    
    return state
    
def show_results(state: AgentState):
    """Show the results of the agent's response."""
    # Get the last assistant message
    last_assistant_message = next((m for m in reversed(state["messages"]) if m["role"] == "assistant"), None)

    if last_assistant_message:
        print("Agent Response:")
        print(last_assistant_message["content"])
    else:
        print("No response from the agent.")
    return state

# Build the complete agent
def build_medical_agent():
    """Build a simplified medical agent with a single node."""
    
    graph = StateGraph(AgentState)
    
    graph.add_node("init_llms", init_llms)
    graph.add_node("input_node", create_input_node)
    graph.add_node("response_node", create_response_node)
    graph.add_node("show_results", show_results)
    # Set the entry point and edge
    graph.set_entry_point("init_llms")
    graph.add_edge("init_llms", "input_node")
    graph.add_edge("input_node", "response_node")
    graph.add_edge("response_node", "show_results")
    # Compile the graph
    return graph.compile()