#%%
"""
本文件用于建立一个中心化监管架构的langGraph Agent，通过中心化监管者架构做调度三个智能体：普通rag agent、GraphRag Agent和写报告workflow Agent
pip install langgraph_supervisor
pip install numpy<2.0
"""
import os
import sys
from config import PROJECT_ROOT, load_project_env

sys.path.insert(0, str(PROJECT_ROOT))
load_project_env()
from Naive_RAG.rag_agent import rag_agent
from GraphRAG.graphrag_agent import graphrag_agent
from langchain.chat_models import init_chat_model
from vlm.multi_model_agent import multi_model_agent
from typing import Literal
from typing_extensions import TypedDict,Annotated    # 如果需要向后兼容
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
# 定义状态
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_node: Literal["rag_agent", "graphrag_agent", "multi_model_agent","__end__"] #
    generate_times: int


model=init_chat_model(model="deepseek-chat", model_provider="deepseek", temperature=0)
from langchain_core.prompts import ChatPromptTemplate

# 分诊智能体prompt
ywfl_prompt = ChatPromptTemplate.from_messages([
("system", """你是一个智能路由助手，负责判断用户问题应该由哪个咨询专家处理，并返回专家名称。如果当前用户的问题与各专家智能体的工作领域不想关，请自己回答
可供选择的智能体：
1. rag_agent：中国内陆(非港澳台地区)旅游基础问题咨询专家
2. graphrag_agent：中国内陆(非港澳台地区)旅游高级问题咨询专家
3. multi_model_agent：港澳台旅游问题咨询专家

说明：
- 当用户希望了解中国内陆(非港澳台地区)各省市景点做有关旅行规划、景点介绍、出发准备、交通线路、相关攻略与书籍、天气情况、宗教文化、住宿、购物、餐饮、当地方言等基础性内容咨询时，请返回:rag_agent
- 当用户希望对中国内陆(非港澳台地区)各省市景点旅游做比较分析、总结概括、关联多个景点、做主题分析、处理概念关联等内容的咨询时，请返回:graphrag_agent
- 当用户希望对香港、澳门、台湾做旅游咨询时，请返回: multi_model_agent
注意:你只需要返回专家名称(rag_agent,graphrag_agent,multi_model_agent),请勿做额外解释说明。

举例1:
输入:历史问题:香港有什么好玩的?\n本轮问题:北京有哪些好玩的景点？
输出:rag_agent
举例2:
输入:历史问题:南昌和浙江比\n本轮问题:饮食文化有什么不同？
输出:graphrag_agent
举例3:历史问题:\n本轮问题:北京出发有哪些景点可以去?
输出:graphrag_agent
举例3:
输入:历史问题:山东什么好玩的?\n本轮问题:香港地区有什么美食推荐
输出:multi_model_agent
举例4:广西的特产有哪些?\n本轮问题:台北有什么好玩的地方?
输出:multi_model_agent

------------------
当前任务:
"""), #、multi_model_agent
("human", "输入:{question}")])
from pydantic import BaseModel,Field
class rag_method_selection(BaseModel):
    rag_method: str = Field(description="rag method:'rag_agent','graphrag_agent','multi_model_agent'") # or 'multi_model_agent' ,'langchain_agent'

#定义转向函数 MessagesState
from typing import Dict,Any
async def check_rag_method(state:State) -> Dict[str, Any]: #,"multi_model_agent"
    print("进入check_rag_method节点:",state["messages"])
    #获取用户最近一次问题
    question_list=[]
    question=None
    history_question_str="历史问题:"
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            question=message.content
            question_list.append(question)
    #最后一个元素为用户本轮问题
    current_query=question_list[-1]
    print(f"用户本轮消息:{question}")
    history_question_str= "".join(question_list[-2:-1])
    print("历史2轮次用户问题:",history_question_str)
    prompt = ywfl_prompt.format(question=f"{history_question_str}+\n本轮问题:{current_query}")
    res = await model.with_structured_output(rag_method_selection).ainvoke([
        {"role": "user", "content": prompt}
    ])
    print("res:",res)
    if "graphrag_agent" in res.rag_method:
        return {"next_node": "graphrag_agent", "generate_times": state["generate_times"], "messages": state["messages"]}
    elif "rag_agent" in res.rag_method:
        return {"next_node":"rag_agent","generate_times": state["generate_times"], "messages": state["messages"]}
    else:
        return {"next_node": "multi_model_agent","generate_times": state["generate_times"]}

#%%
def create_workflow():
    workflow = StateGraph(State)
    # 添加节点
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("graphrag_agent", graphrag_agent)
    workflow.add_node("multi_model_agent", multi_model_agent)
    workflow.add_node("rag_router", check_rag_method)  # 将路由函数作为节点
    # 设置入口点START->rag_router
    workflow.set_entry_point("rag_router")
    # 根据rag_router判断结果路由到下游的agent中处理
    workflow.add_conditional_edges(
        "rag_router",
        lambda x: x["next_node"],  # check_rag_method 返回 {"next": "xxx"}
        {
            "rag_agent": "rag_agent",
            "graphrag_agent": "graphrag_agent",
            "multi_model_agent": "multi_model_agent"
        }
    )
    # 设置结束
    workflow.add_edge("rag_agent", END)
    workflow.add_edge("graphrag_agent", END)
    workflow.add_edge("multi_model_agent", END)
    return workflow.compile()
compiled_workflow=create_workflow()
print("总智能体流程:",compiled_workflow)

def save_workflow_graph(output_path=PROJECT_ROOT / "build_system_graph.png"):
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
    from io import BytesIO

    graph_png = compiled_workflow.get_graph().draw_mermaid_png()
    img = mpimg.imread(BytesIO(graph_png), format='PNG')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    with open(output_path, "wb") as f:
        f.write(graph_png)

#%%

messages_list=[] #用于存储历史对话,生产环境下只需要将这个历史对话记忆存储在对应的数据库中即可。每次去读取这个用户的数据的数据message_list传入即可

def safe_input() -> str:
    raw = sys.stdin.buffer.readline()
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace").strip()


async def run_workflow():
    while True:
        print("\n请输入本轮问题（输入 'exit' 退出）:")
        user_input = safe_input()
        if user_input.lower() == "exit":
            print("退出对话...")
            break
        if not user_input:
            print("问题不能为空，请重新输入！")
            continue
        # 新增人类消息
        human_msg = HumanMessage(content=user_input)
        messages_list.append(human_msg)
        # 重置初始化状态和生成次数统计值
        initial_state = {
            "messages": messages_list,
            "next_node": "",
            "generate_times": 0
        }
        try:
            # 调用 workflow 并捕获详细异常
            result = await compiled_workflow.ainvoke(initial_state)
            # 提取 AI 回复
            final_ai_msg = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    final_ai_msg = msg
                    break
            if final_ai_msg:
                messages_list.append(final_ai_msg)
                print(f"本轮回复结果: {final_ai_msg.content}")
            else:
                print("未获取到有效回复")
        except Exception as e:
            # 打印完整异常堆栈（关键！）
            import traceback
            print(f"执行错误详情: {e}")
            traceback.print_exc()
            # 回滚：移除本次的人类消息，避免污染历史
            if human_msg in messages_list:
                messages_list.remove(human_msg)
            continue

import asyncio

if __name__ == "__main__":
    if os.getenv("DRAW_AGENT_GRAPHS") == "1":
        save_workflow_graph()
    asyncio.run(run_workflow())
#%%
# graphrag_agent:从上海出发有哪些旅游景点可以去？
# multi_model_rag_agent:香港有什么好玩的地方？
# rag_agent:从大理坐大巴到双廊需要多久？
# rag_agent+多轮对话: 到了之后住哪里?

