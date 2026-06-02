"""
注意：当前构建的graphrag_agent是不带有任何记忆的，如果要连续对话继承记忆，需要做响应的工程化改造
"""
#%%
#pip install azure-storage-blob
from __future__ import annotations
import os
import asyncio
from pathlib import Path
import sys
from typing import Literal, Dict, Any
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool
from langchain.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
# 加载GraphRAG数据和配置
from graphrag.config.load_config import load_config
import graphrag.api as api
from typing import Literal
from typing_extensions import TypedDict,Annotated    # 如果需要向后兼容
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, END, add_messages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GRAPHRAG_OUTPUT_DIR, GRAPHRAG_ROOT, PROJECT_ROOT, get_env, load_project_env, require_env, require_path

load_project_env()

_graphrag_resources = None


def read_graphrag_parquet(name: str) -> pd.DataFrame:
    path = require_path(
        GRAPHRAG_OUTPUT_DIR / name,
        "GraphRAG index file. Run `graphrag index --root GraphRAG/tourist_graphrag` first",
    )
    return pd.read_parquet(path)


def read_optional_graphrag_parquet(name: str) -> pd.DataFrame | None:
    path = GRAPHRAG_OUTPUT_DIR / name
    if not path.exists():
        return None
    return pd.read_parquet(path)


def get_graphrag_resources():
    global _graphrag_resources
    if _graphrag_resources is None:
        cfg = load_config(require_path(GRAPHRAG_ROOT, "GraphRAG root"))
        _graphrag_resources = {
            "cfg": cfg,
            "entities": read_graphrag_parquet("entities.parquet"),
            "communities": read_graphrag_parquet("communities.parquet"),
            "community_reports": read_graphrag_parquet("community_reports.parquet"),
            "text_units": read_graphrag_parquet("text_units.parquet"),
            "relationships": read_graphrag_parquet("relationships.parquet"),
            "covariates": read_optional_graphrag_parquet("covariates.parquet"),
        }
    return _graphrag_resources
#%%
MODEL_NAME = "deepseek-chat"
model_config = {
    "model": MODEL_NAME,
    "model_provider": "deepseek",
    "api_key": require_env("DEEPSEEK_API_KEY"),
    "base_url": get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
}
# print(**model_config)
#%%
from langchain.chat_models import init_chat_model
model=init_chat_model(model="deepseek-chat",
                        model_provider="deepseek",
                        api_key=require_env("DEEPSEEK_API_KEY"),
                        base_url=get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
if os.getenv("DEBUG_GRAPHRAG_AGENT") == "1":
    print(model)

#%%
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    generate_times: int
#%%
# ---------------------------------------------------------------------------
# GraphRAG Retriever Tool:SYSTEM_INSTRUCTION会指导当需要提供课程材料的时候进行调用 ----------------------------------------------------
# # ---------------------------------------------------------------------------
@tool
async def national_graphrag_retrieve_tool(query: str) -> str:
    """
    调用GraphRAG进行知识库检索，根据query检索相关有用内容。
    """
    resources = get_graphrag_resources()
    try:
        response, _ = await api.local_search(
            config=resources["cfg"],
            entities=resources["entities"],
            communities=resources["communities"],
            community_reports=resources["community_reports"],
            text_units=resources["text_units"],
            relationships=resources["relationships"],
            covariates=resources["covariates"],
            community_level=1,
            response_type="Detailed answer with attraction names, city names, transport routes, and concrete recommendations",
            query=query,
        )
        if response and str(response).strip():
            print("graphrag_local_search结果:", response)
            return str(response)
    except Exception as exc:
        print("graphrag_local_search failed, fallback to global_search:", repr(exc))

    response, _ = await api.global_search(
        config=resources["cfg"],
        entities=resources["entities"],
        communities=resources["communities"],
        community_reports=resources["community_reports"],
        community_level=1, #社区划分的层级深度,值越大表示更细粒度的社区划分,level=2 表示中等粒度的社区结构
        dynamic_community_selection=False, #False
        response_type="Detailed answer with concrete attraction names, city names, and travel recommendations",
        query=query,
    )
    print("graphrag_search结果:",response)
    return response

# response=await national_graphrag_retrieve_tool(query="从长沙出发有哪些景点") #从长沙出发有哪些景点 广西桂林有什么好玩的景点
# print(response)
#%%
SYSTEM_INSTRUCTION = """
你是一个中国大陆旅游向导，你善于回答回答中国大陆各省市旅游(非港澳台地区)相关的一切问题。
包括：中国大陆各省市旅游的旅行规划、景点介绍、出发准备、交通线路、相关攻略与书籍、天气情况、宗教文化、住宿、购物、餐饮、当地方言等。
如果用户的问题与中国大陆各省市旅游无关，请回复：我无法回答与中国各省市旅游无关的问题。
技能：遇到中国大陆各省市旅游的问题，你必须调用national_graphrag_retrieve_tool进行知识库检索知识进行回复。
请注意：请勿回答除中国大陆省市旅游之外的问题，请勿回答港澳台地区旅游相关的问题
"""
GRADE_PROMPT = """你是一位评估检索到的文档与用户问题相关性的评分员。\n
检索到的文档：\n{context}\n\n用户问题：{question}\n"
若相关请返回'yes'，否则返回'no'"""
#
REFINE_PROMPT="""你的任务是优化或澄清用户的问题，使得修改后的问题能更符合中国大陆旅游相关主题。
注意：
- 中国大陆旅游相关指的是例如：旅行规划、景点介绍、出发准备、交通线路、相关攻略与书籍、天气情况、宗教文化、住宿、购物、餐饮、当地方言等。
- 你只需要输出优化后的用户问题，请勿做多余解释。
- 若用户本轮问题表述已经完整请勿结合历史会话进行问题融合,仅对本轮问题进行改写;若用户本轮问题表述不完整,需要结合历史问题进行补充式改写
举例1:
原始问题:历史问题:大连旅游\n本轮问题:有什么好玩的地方 
优化后问题:去大连旅游什么好玩的地方推荐
举例2:
原始问题:历史问题:青岛有什么好吃的\n本轮问题:秦皇岛有什么好玩的? 
优化后问题:秦皇岛有什么好玩的景点? 
举例3:
原始问题:历史问题:北京故宫的旅游攻略?\n本轮问题:颐和园怎么去比较方便?
优化后问题:颐和园的交通线路有哪些?

------
原始问题:{question}
优化后问题:"""

ANSWER_PROMPT ="""您是中国内陆各省市旅游智能助手。请严格依据为您检索到的'参考文档'，回答用户提出的旅游相关咨询。
### 回答原则
1.答案必须完全依据所提供的上下文信息，不编造、不臆测，确保信息完整且准确。
2.特别对于港澳台旅行规划、景点介绍、出发准备、交通线路、相关攻略与书籍、天气情况、宗教文化、住宿、购物、餐饮、当地方言等内容，请务必使用上下文中的材料。
3. 如果上下文信息无法解答用户问题，请直接回复：“我不知道。”

### 本轮任务：
输入:
用户问题：{question}
参考文档：{context}
输出:"""

# LangGraph节点定义
async def generate_respond(state: State):
    """LLM decides to answer directly or call GraphRAG tool."""
    state["generate_times"]+=1 #统计检索次数
    print("进入GraphRag_agent,当前回复次数+1:", state["generate_times"])
    response = await model.bind_tools([national_graphrag_retrieve_tool]).ainvoke(
        [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            *state["messages"],
        ]
    )
    print("generate_respond", response)
    return {"generate_times":state["generate_times"],"messages": [response]}


class GradeDoc(BaseModel):
    relevant_score: str = Field(description="如果是用户问题的答案输出:'yes',如果没有检索文档或与用户问题无关或没有解决用户问题输出:'no'.")

async def grade_search_docs(state: State, evaluate_model=model) -> Literal[
    "generate_final_answer", "refine_query"]:
    # question = state["messages"][0].content  # original user question
    #获取用户最近一次question
    question=None
    for message in state["messages"][::-1]:
        if isinstance(message, HumanMessage):
            question=message.content
            print(f"这是最近一次用户消息:{question}")
            break
    if not question:
        return "generate_final_answer"
    ctx = state["messages"][-1].content  # retriever output
    # 我不能回答与 MCP 技术实战公开课无关的问题
    print("进入Graphrag_agent grade_document节点", state)
    if state['generate_times']>2:
        print("进入Graphrag_agent打分节点,当前生成次数超过最大次数2次，不进行改写自由生成")
        return 'generate_final_answer'
    print(
        f"进入Graphrag_agent打分节点,当前生成次数{state['generate_times']}，"
        f"改写输入question:{question},context:{ctx}"
    )
    prompt = GRADE_PROMPT.format(question=question,context=ctx)
    print("prompt:", prompt)
    # 第二次调用大模型做第一次调用大模型输出的格式整理,整理成"Relevance score 'yes' or 'no'." 得到检索得到的结果的相关性评估结果
    result = await evaluate_model.with_structured_output(GradeDoc).ainvoke([  # with_structured_output(GradeDoc).
        {"role": "user", "content": prompt}
    ])
    print("result:", result)
    if not result:
        return "refine_query"
    # 如果没有成功解析出对应的数据结构则也让模型自由生成答案
    return "generate_final_answer" if (result.relevant_score.lower().startswith("y")) else "refine_query"


async def refine_query(state: State):
    # question = state["messages"][0].content
    question=None
    for message in state["messages"][::-1]:
        if isinstance(message, HumanMessage):
            question=message.content
            print(f"这是最近一次用户消息:{question}")
            break
    if not question:
        return {"generate_times": state["generate_times"], "messages": [{"role": "user", "content": "改写出错"}]}
    print("进入GraphRag agent重写节点:", state)
    prompt = REFINE_PROMPT.format(question=question)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    return {"generate_times":state["generate_times"],"messages": [{"role": "user", "content": resp.content}]}

async def generate_final_answer(state: State):
    # question = state["messages"][0].content
    question=""
    for message in state["messages"][::-1]:
        if isinstance(message, HumanMessage):
            question=message.content
            print(f"这是最近一次用户消息:{question}")
            break
    print("进入GraphRag agent生成回答节点:", state)
    print("question:", question)
    ctx = state["messages"][-1].content #取出tool_message
    if ctx and str(ctx).strip():
        return {"messages": [AIMessage(content=str(ctx))]}
    prompt = ANSWER_PROMPT.format(question=question, context=ctx)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [resp]}

workflow = StateGraph(State)
workflow.add_node("generate_respond", generate_respond)
workflow.add_node("retrieve", ToolNode([national_graphrag_retrieve_tool])) #定义工具节点
workflow.add_node("refine_query", refine_query)
workflow.add_node("generate_final_answer", generate_final_answer)
workflow.add_edge(START, "generate_respond")
#根据是否工具调用选择路由,如果有工具调用则需要对调用结果grade_search_docs评分是否相关
workflow.add_conditional_edges("generate_respond", tools_condition, {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_search_docs)
workflow.add_edge("generate_final_answer", END)
workflow.add_edge("refine_query", "generate_respond")
graphrag_agent = workflow.compile(name="graphrag_agent")
print("Graphrag Agent构建完成")
#%%
#绘制langGraph流程图
if os.getenv("DRAW_AGENT_GRAPHS") == "1":
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
    from io import BytesIO
# 打印文本流程图
    graph_png = graphrag_agent.get_graph().draw_mermaid_png()
    img = mpimg.imread(BytesIO(graph_png), format='PNG')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    with open(PROJECT_ROOT / "build_graphrag.png", "wb") as f:
        f.write(graph_png)
#%%
#从济南出发有哪些景点推荐去
#福州出发2小时以内有哪些地方可以去
#中国有哪些有名的"山"类型的景点
# # 运行异步函数
# async def chat_with_agent():
#     print("GraphRAG 旅游助手已启动！输入 '退出' 结束对话")
#     print("=" * 50)
#     while True:
#         try:
#             user_input = input("\n请输入您的问题: ").strip()
#             if user_input.lower() in ['退出', 'exit', 'quit']:
#                 print("感谢使用，再见！")
#                 break
#
#             if not user_input:
#                 continue
#
#             print("\n正在处理...")
#             result = await graphrag_agent.ainvoke({
#                 "messages": [{"role": "user", "content": user_input}],
#                 "generate_times": 0
#             })
#             response = result.get("messages")[-1].content
#             print(f"\n助手回答: {response}")
#
#         except KeyboardInterrupt:
#             print("\n\n对话已中断")
#             break
#         except Exception as e:
#             print(f"\n发生错误: {str(e)}")
#
#
# if __name__ == "__main__":
#     asyncio.run(chat_with_agent())
