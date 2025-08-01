FILL_IN_FORM_PROMPT = """
我将给你一段医生诊断报告经过OCR提取之后的文本。基于以下文本，判断是否在报告中提及了以下关键信息。如果提及了，请从文本中总结或提取出以下内容并返回给我，如果没有提及，请返回NO给我
**医疗诊断报告：**
{ocr_text}
**关键信息/指标：**
{key_info}
**返回格式：**
{{
"key_1": "value_1",
"key_2": "value_2",
...
}}
**返回示例：**
假设我有如下关键信息需要提取：
[LM, LAD]
并且报告中包含了如下信息：
“左主干(LM) 管壁见钙化斑块，管腔轻度狭窄。左前降支(LAD)见近中段钙化斑块，局部管腔重度狹窄；”
那么我可能会返回：
{{
"LM": "管壁见钙化斑块",
"LAD": "近中段钙化斑块，局部管腔重度狹窄"
}}
**输出：**
"""


FILLIN_PROMPT_2 = """
我将给你一段医生诊断报告经过OCR提取之后的文本。基于以下文本，判断该患者的“冠状动脉起源、走形及终止”是正常还是异常。请回复正常或者异常，并简要给出理由。
**医疗诊断报告：**
{ocr_text}
**返回格式：**
{{
"key_name": "冠状动脉起源、走形及终止", # 固定
"result": "value_1",    # 正常或者异常
"reason": "value_2"    # 理由
}}
**返回示例：**
{{
"key_name": "冠状动脉起源、走形及终止",
"result": "正常",
"reason": "替换成你认为的根据报告产生的合适的理由"
}}
**输出：**
"""


FILLIN_PROMPT_3 = """
我将给你一段医生诊断报告经过OCR提取之后的文本。基于以下文本，判断该患者的属于“右冠优势型”， “左冠优势型”， 还是“均衡型”。请回复判断结果，并简要给出理由。请返回标准JSON格式。
**医疗诊断报告：**
{ocr_text}
**返回格式：**
{{
"key_name": "冠脉优势型",  # 固定
"result": "value_1",    # “右冠优势型”， “左冠优势型”， 还是“均衡型”
"reason": "value_2"    # 理由
}}
**返回示例：**
{{
"key_name": "冠脉优势型",
"result": "右冠优势型",
"reason": "替换成你认为的根据报告产生的合适的理由"
}}
**输出：**
"""


FILLIN_PROMPT_4 = """
我将给你一段医生诊断报告经过OCR提取之后的文本。基于以下文本，提取“异常描述”。如果没有任何异常描述，请返回“NO”给我。请返回标准JSON格式。
**医疗诊断报告：**
{ocr_text}
**返回格式：**
{{
"key_name": "异常描述",  # 固定
"result": "value_1",    # 异常描述
}}
**输出：**
"""


FILLIN_PROMPT_5 = """
我将给你一段医生诊断报告经过OCR提取之后的文本。请你基于以下文本，提取冠脉节段"{location}(名称)"的以下字段信息：
- 斑块种类
- 类型
- 症状
- 数值
- 狭窄程度
- 闭塞

如果相应字段没有提及，请返回"-"（或"否"等合适的默认值）。

**字段说明：**
- "闭塞"字段请返回"是"或者"否"；
- "斑块种类"可选：软斑块（非钙化性斑块）、混合密度斑块、硬斑块（钙化性斑块）；
- "狭窄程度"可选：局限性狭窄（长度＜10mm）、阶段性狭窄（10-20mm）、弥漫性狭窄（＞20mm）；
- 其余字段如未提及请填写"-"。

**医疗诊断报告：**
{ocr_text}

**返回格式：**
{{
"斑块种类": "value_斑块种类",
"类型": "value_类型",
"症状": "value_症状",
"数值": "value_数值",
"狭窄程度": "value_狭窄程度",
"闭塞": "value_闭塞"
}}

**只返回JSON，不要输出其他内容。**
"""


REPORT_CLASSIFIER_PROMPT = """
我将给你一段医疗报告经过OCR提取之后的文本。请你判断这份报告主要是关于什么类型的检查。

请仔细阅读报告内容，然后判断它是：
1. **冠脉CTA报告** - 主要描述冠状动脉血管结构、斑块、狭窄程度等
2. **心脏超声报告** - 主要描述心脏腔室尺寸、心功能、血流速度等生理功能指标

**医疗诊断报告：**
{ocr_text}

**判断依据：**
- 如果报告中主要包含"左主干"、"前降支"、"回旋支"、"右冠"、"斑块"、"狭窄"、"钙化"等词汇，则为冠脉CTA报告
- 如果报告中主要包含"射血分数"、"舒张期"、"收缩期"、"房室腔"、"瓣膜"、"血流速度"、"多普勒"等词汇，则为心脏超声报告

**返回格式：**
{{
"report_type": "CTA",    # 只能是 "CTA" 或 "Ultrasound"
"confidence": "高",      # 置信度：高/中/低
"reason": "判断理由"     # 简要说明判断依据
}}

**只返回JSON，不要输出其他内容。**
"""


ULTRASOUND_EXTRACT_PROMPT = """
我将给你一段心脏超声诊断报告经过OCR提取之后的文本。请你基于以下文本，**仅为超声测量项目"{location}"**提取以下字段信息：
- 斑块种类
- 类型  
- 症状
- 数值
- 狭窄程度
- 闭塞

**重要说明：**
- 请**严格按照给定的项目名称"{location}"进行精确匹配**
- 只提取与"{location}"完全对应的数值，不要提取其他相似项目的数值
- 对于超声心动图测量项目，只有"数值"和"闭塞"字段有意义
- "斑块种类"、"类型"、"症状"、"狭窄程度" 字段对超声测量不适用，请始终返回 "-"
- "数值"字段：提取具体的测量数值（仅数字部分，不包含单位）
- "闭塞"字段：对于血管相关测量返回"是"或"否"，其他测量返回"否"

**严格要求：**
- 如果报告中没有提到项目名称"{location}"的确切信息，请在"数值"字段返回"-"
- 不要使用任何别名或相似词汇进行匹配
- 不要将其他项目的数值错误分配给当前项目

{dynamic_alias_rules}

**医疗诊断报告：**
{ocr_text}

**返回格式：**
{{
"斑块种类": "-",
"类型": "-", 
"症状": "-",
"数值": "实际测量数值或-",
"狭窄程度": "-",
"闭塞": "否"
}}

**只返回JSON，不要输出其他内容。**
"""
