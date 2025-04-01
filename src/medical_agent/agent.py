from typing import List, Dict, Any, TypedDict, Literal, Union
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
import base64
from pathlib import Path
from openai import OpenAI
import os
from medical_agent.utils import call_qwen_vl_api, safe_json_load
from medical_agent.utils import *
from medical_agent.table_format import create_formatted_df, ROW_INDEX
from typing import TypedDict, get_type_hints, Any
from medical_agent.prompts import FILL_IN_FORM_PROMPT, FILLIN_PROMPT_2, FILLIN_PROMPT_3, FILLIN_PROMPT_4, FILLIN_PROMPT_5
import json
from medical_agent.gui import show_popup_with_df

# Define message types
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: Union[str, Dict[str, Any]]
    content_type: Literal["text", "image"] = "text"


# Define the system prompt
SYSTEM_PROMPT = """你是一个医疗助手，你的主要任务是帮助医生整理病例，诊断报告等文件或图片，并将其整理成结构化数据存储起来。必要时，你也可以回答用户的医疗问题。如果可能，在回答医疗问题时请尽量提供来源。"""

class AgentState(TypedDict):
    messages: list
    qwen: Any
    gpt: Any
    formatted_table: Any
    row_index: dict
    image_content: dict
    context: dict
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

    state['formatted_table'] = create_formatted_df()
    state['row_index'] = ROW_INDEX

    return state

# Define the response generation node
def create_input_node(state: AgentState):
    """Create a node for handling user input and generating responses."""
    
    # Default values
    image_path = "/Users/czy/Desktop/sidegig/Medical_Agent/data/input_1.jpg"
    question = "请分析这张医疗图像并提供诊断建议。"
    
    # Get user input or use defaults
    # DEBUG 模式下跳过输入
    if os.environ.get('DEBUG', '0') == '0':
        try:
            image_path = input("请输入图片路径 (直接回车使用默认路径): ").strip() or default_image_path
            question = input("请输入您的问题 (直接回车使用默认问题): ").strip() or default_question
        except:
            pass
    
    # Read and encode the image
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"警告：找不到图片 {image_path}，请重新输入")
            raise RuntimeError(f"找不到图片 {image_path}")
            
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Create image content in the format expected by the API
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }

        state['image_content'] = image_content
        
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


def ocr_node(state: AgentState):
    """Create a node for OCR."""
    image_content = state['image_content']
    if not image_content:
        print("Warning: There is no image content to process. Skipping OCR.")
        return state

    # Call the OCR API
    messages=[
        {
            "role": "user",
            "content": [
                image_content,
                # 为保证识别效果，如果使用qwen-vl-ocr 系列模型， 目前模型内部会统一使用"Read all the text in the image."进行识别，用户输入的文本不会生效。
                {"type": "text", "text": "Read all the text in the image"},
            ],
        }
    ]

    print("正在调用OCR模型获取文本提取结果")

    if os.environ.get('DEBUG', '0') == '1':
        text = """CT检查报告单
(扫码查看图像
检查号：220901
申请科室：内分泌科Ⅱ病区
申请医生：张玲
姓名： 住院号：2022041478 年龄：55岁 性别：男 床号：5
检查项目：256排冠状动脉CTA
检查所见：
冠状动脉呈右优势型。左主干起源于左窦，右冠状动脉起源于右窦。
左主干管壁可见钙化斑块，管腔轻微狭窄约10%。左前降支近段管壁可见钙化斑块，管腔轻度狭窄约25%；中段管壁可见混合斑块，管腔重度狭窄约85%；远段管壁可见钙化斑块，管腔轻度狭窄约25%。第一，第二对角支未见斑块及明显狭窄。左回旋支中远段管壁可见非钙化斑块，管腔轻度狭窄约25%；近段未见斑块及明显狭窄。第一，第二钝缘支未见斑块及明显狭窄。中间支未见斑块及明显狭窄。
右冠状动脉近段管壁可见钙化、非钙化斑块，管腔轻度狭窄约25%；中段、远段未见斑块及明显狭窄。右后降支未见斑块及明显狭窄。左室后支未见斑块及明显狭窄。
心脏各腔室不大，心肌未见异常密度影。
印象：
冠状动脉CTA：1.左主干管壁钙化斑块，管腔轻微狭窄。2.左前降支近段管壁钙化斑块，管腔轻度狭窄；中段管壁混合斑块，管腔重度狭窄；远段管壁钙化斑块，管腔轻度狭窄。3.左回旋支中远段管壁非钙化斑块，管腔轻度狭窄。4.右冠状动脉近段管壁钙化、非钙化斑块，管腔轻度狭窄。
检查日期：2022-09-08 审核医师：
报告医师：
注：1.本报告仅供临床科室申请医生诊治参考！
2.二维码链接图像，请妥善保存本报告！
报告时间：2022-09-08 17:19:39
        """
    else:
        completion = state["qwen"].chat.completions.create(
            model="qwen-vl-ocr",
            messages=state['messages']
        )
        text = completion.choices[0].message.content


    state['context']['ocr'] = text
    return state


def fill_form_node(state: AgentState):
    """Create a node for filling the form."""
    # Get the last assistant message
    ocr = state['context']['ocr']
    formatted_table = state['formatted_table']
    row_index = state['row_index']
    # Call the LLM to fill the form
    header_data = [
        ["冠状动脉钙化总积分", "LM", "LAD", "LCX", "RCA"]
    ]
        #     ["【冠状动脉起源、走形及终止】", "正常", "异常", "右冠优势型", "左冠优势型", "均衡性"],
        # ["异常描述"]

    # SECTION BEGIN: 填充表格头部部分
    top_data = {}
    header_output = []
    qwen = state["qwen"]
    for row in header_data:
        input_prompt = FILL_IN_FORM_PROMPT.format(
            ocr_text=ocr,
            key_info=row,
        )
        completion = qwen.chat.completions.create(
            model="qwen-max-0125",
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': input_prompt}]
        )
        text = completion.choices[0].message.content
        tmp = safe_json_load(text)
        header_output.append(tmp)
        for k in tmp:
            top_data[k] = tmp[k]
            if top_data[k] == "NO":
                top_data[k] = ""
        print(text)
    input_prompt_list = []
    input_2 = FILLIN_PROMPT_2.format(ocr_text=ocr)
    input_3 = FILLIN_PROMPT_3.format(ocr_text=ocr)
    input_4 = FILLIN_PROMPT_4.format(ocr_text=ocr)

    input_prompt_list = [input_2, input_3, input_4]
    for input_prompt in input_prompt_list:
        completion = qwen.chat.completions.create(
            model="qwen-max-0125",
            messages=[
                {'role':'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': input_prompt}]
        )
        text = completion.choices[0].message.content
        tmp = safe_json_load(text)
        header_output.append(tmp)
        top_data[tmp['key_name']] = tmp['result']
        print(text)


    
    # SECTION END --------

    SECTION BEGIN: 填充表格主体部分
    for i in range(len(formatted_table)):
        location = formatted_table.iloc[i]["冠脉节段"]
        if location in row_index:
            ridx = row_index[location]
            input_prompt = FILLIN_PROMPT_5.format(
                ocr_text=ocr,
                location=location,
            )
            completion = qwen.chat.completions.create(
                model="qwen-max-0125",
                messages=[
                    {'role':'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': input_prompt}]
            )
            text = completion.choices[0].message.content
            print(f"result for {location} is {text}")

            tmp = safe_json_load(text)
            for col in list(formatted_table.columns)[1:]:
                value = tmp.get(col, "")
                if value == "NO":
                    value = ""
                formatted_table.at[ridx, col] = value
            save_df_to_cache(formatted_table, "qwen_cache")

    save_df_to_cache(formatted_table, "qwen_cache")

    # SECTION END --------

    # SECTION BEGIN: Load df from Cache 只在debug的时候使用
    df = load_df_from_cache("qwen_cache")
    state['context']['df'] = df
    show_popup_with_df(df, top_data)

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
    graph.add_node("ocr_node", ocr_node)
    graph.add_node("response_node", create_response_node)
    graph.add_node("show_results", show_results)
    graph.add_node("fill_form_node", fill_form_node)
    # Set the entry point and edge
    graph.set_entry_point("init_llms")
    graph.add_edge("init_llms", "input_node")
    graph.add_edge("input_node", "ocr_node")
    graph.add_edge("ocr_node", "fill_form_node")
    # graph.add_edge("ocr_node", "response_node")
    # graph.add_edge("response_node", "show_results")
    # Compile the graph
    return graph.compile()