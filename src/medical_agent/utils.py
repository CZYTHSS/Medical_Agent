def call_qwen_vl_api(state):
    result = state["qwen"].chat.completions.create(
        model="qwen-vl-max-latest",
        messages=state['messages']
    )

    text = completion.choices[0].message.content
    return text


