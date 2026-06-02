from __future__ import annotations
import os
import asyncio
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FAISS_INDEX_DIR, PROJECT_ROOT, load_project_env, require_env, require_path

load_project_env()
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
os.environ["DEEPSEEK_API_KEY"] = require_env("DEEPSEEK_API_KEY")
MODEL_NAME = "deepseek-chat"
model = init_chat_model(model=MODEL_NAME, model_provider="deepseek",temperature=0)
#%%
from langchain_community.embeddings import DashScopeEmbeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3", dashscope_api_key=require_env("QWEN_API_KEY")
)
rag_vectordb = require_path(FAISS_INDEX_DIR, "Naive RAG FAISS index")
vector_store = FAISS.load_local(
    folder_path=str(rag_vectordb),
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)
#%%
# rag_retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
rag_retriever=vector_store.as_retriever(search_type="similarity_score_threshold",
                                    search_kwargs={"score_threshold":0.3,"k": 10})

# 定义一个知识库检索工具,当询问mcp相关问题时进行调用
national_retriever_tool = create_retriever_tool(
    rag_retriever,
    name="national_retriever_tool",
    description="查询并返回中国内陆(非港澳台)各省市旅游相关的文档",
)
# vector_result=vector_store.similarity_search(query="黄姚古镇旅游线路推荐一下",kwargs={"score_threshold":0.1})
# print("vector_result:",vector_result)
tool_result=national_retriever_tool.invoke("从兰州出发，车程约两小时可到达哪些旅游景点？") #黄姚古镇旅游线路推荐一下
print("tool_result:",tool_result)
# %%
SYSTEM_INSTRUCTION = """
你是一个中国大陆旅游向导，你善于回答中国内陆各省市旅游相关的一切问题。
包括：中国大陆各省市旅游的旅行规划、景点介绍、出发准备、交通线路、相关攻略与书籍、天气情况、宗教文化、住宿、购物、餐饮、当地方言等。
如果用户的问题与中国大陆各省市旅游无关，请回复：我无法回答与中国各省市旅游无关的问题。
技能：遇到中国大陆各省市旅游的问题，你必须调用national_retrieve_tool进行知识库检索知识进行回复。
如果检索的知识不能支持回复用户问题，请回复：我不知道
请注意：请勿回答除中国大陆省市旅游之外的问题，请勿回答港澳台地区旅游相关的问题。
"""
# 评估检索得到的结果是否与用户的问题相关,输出yes或者no。
#评估提示词
GRADE_PROMPT = """你是一位评估检索到的文档是否能解决用户问题的评分员。\n
检索到的文档：\n{context}\n\n用户问题：{question}\n"
若将用户问题成功解决请返回'yes'，若需要用户提供更多信息或无法明确给出答案请返回'no'""" #相关
#
REFINE_PROMPT="""你的任务是优化或澄清用户的问题，使得修改后的问题能更符合中国大陆旅游相关主题。
注意：
- 中国大陆旅游相关指的是例如：旅行规划、景点介绍、出发准备、交通线路、相关攻略与书籍、天气情况、宗教文化、住宿、购物、餐饮、当地方言等。
你只需要输出优化后的用户问题，请勿做多余解释。
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

# - 如果用户问题偏离主题，让模型改写问题，使其更贴近“工具调用 / LangGraph / MCP”等关键概念。
# - 给模型一个问题和上下文，引导它用 Markdown、代码块、图片等方式生成结构化答案。
ANSWER_PROMPT = """您是一位专业的中国内陆旅游智能助手。请严格依据为您检索到的'参考文档'，回答用户提出的旅游相关咨询。
### 回答原则
1.答案必须完全依据所提供的上下文信息，不编造、不臆测，确保信息完整且准确。
2.特别对于中国内陆旅行规划、景点介绍、出发准备、交通线路、相关攻略与书籍、天气情况、宗教文化、住宿、购物、餐饮、当地方言等内容，请务必使用上下文中的材料。
3. 如果上下文信息无法解答用户问题，请直接回复：“我不知道。”

### 本轮任务：
输入:
用户问题：{question}
参考文档：{context}
输出:"""
from langgraph.graph import StateGraph, END
from typing import Literal
from typing_extensions import TypedDict,Annotated    # 如果需要向后兼容
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    generate_times: int

# 定义langgraph节点
async def generate_response(state: State):
    print("rag_state:",state)
    state["generate_times"]+=1 #统计检索次数
    print("进入rag_agent,当前系统回复次数+1:", state["generate_times"])
    """LLM decides to answer directly or call retriever tool."""
    response = await model.bind_tools([national_retriever_tool]).ainvoke(
        [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            *state["messages"],
        ]
    )
    messages_list=state["messages"]
    messages_list.append(response)
    print("出generate_response的message_list:",messages_list)
    print("generate_response", response)
    return {"generate_times":state["generate_times"],"messages": messages_list} #[response]


# 定义一个输出的str字符串类型,希望输出结果是yes或no，但是不是强制枚举值类型,考虑到模型输出的不稳定性
class GradeDoc(BaseModel):
    relevant_score: str = Field(description="如果是用户问题的答案输出:'yes',如果没有检索文档或与用户问题无关或没有解决用户问题输出:'no'.")


# 如果检索得到的结果,经过评估模型输出结果为相关(y)则到generate_final_answer节点，否则到refine_query节点。
# 这一个评估模块非常的重要,起到一个路由作用：如果相关则直接return_response，如果不相关则路由到问题改写prompt refine_query
# 定义路由函数
async def grade_search_docs(state: State, evaluate_model=model) -> Literal[
    "generate_final_answer", "refine_query"]:
    #获取用户最近一次question
    question=None
    for message in state["messages"][::-1]:
        if isinstance(message, HumanMessage):
            question=message.content
            break
    if not question:
        return "generate_final_answer"
    ctx = state["messages"][-1].content  # 检索+生成的结果
    # 我不能回答与 MCP 技术实战公开课无关的问题
    print("进入rag_agent grade_document节点", state)
    if state['generate_times']>2:
        print("进入rag_agent打分节点,当前生成次数超过最大次数2次，不进行评分改写走自由生成")
        return 'generate_final_answer'
    print(
        f"进入rag_agent打分节点,当前生成次数{state['generate_times']}，"
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


# improve用户问题节点
async def refine_query(state: State):
    print("进入refine_query节点:",state)
    # question = state["messages"][0].content
    recent_user_question_list=[]
    question_count=0
    for message in state["messages"][::-1]:
        #最近用户两轮问题拿过来做拼接用于上下文连贯性query改写,超过2轮不做考虑
        if question_count>2:
            break
        if isinstance(message, HumanMessage):
            question_count+=1 #用户问题+1
            recent_user_question_list.append(message.content)
    #将用户近两轮回复按顺序拼接
    recent_user_question_str="\n".join(recent_user_question_list[::-1])
    print("query refine input:",recent_user_question_str)
    if not recent_user_question_str:
        return {"generate_times": state["generate_times"], "messages": [{"role": "user", "content": "改写出错"}]}
    print("进入rag agent重写节点:", state)
    prompt = REFINE_PROMPT.format(question=recent_user_question_str)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    print("query_refine_result:",resp)
    return {"generate_times":state["generate_times"],"messages": [{"role": "user", "content": resp.content}]}

# 用检索到的知识生成答案节点
async def generate_final_answer(state: State):
    # question = state["messages"][0].content
    for message in state["messages"][::-1]:
        if isinstance(message, HumanMessage):
            print("这是最近一次用户消息")
            question=message.content
            break
    print("进入rag agent生成回答节点:", state)
    print("question:", question)
    ctx = state["messages"][-1].content
    prompt = ANSWER_PROMPT.format(question=question, context=ctx)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [resp]}
#%%
# Build graph
workflow = StateGraph(State)
workflow.add_node("generate_response", generate_response)
workflow.add_node("retrieve", ToolNode([national_retriever_tool]))
workflow.add_node("refine_query", refine_query)
workflow.add_node("generate_final_answer", generate_final_answer)
workflow.add_edge(START, "generate_response")
workflow.add_edge("generate_response", "retrieve")
workflow.add_conditional_edges("retrieve", grade_search_docs)
workflow.add_edge("generate_final_answer", END)
workflow.add_edge("refine_query", "generate_response")
rag_agent = workflow.compile(name="rag_agent")
print("rag Agent构建完成")
def save_workflow_graph(output_path=PROJECT_ROOT / "build_graph.png"):
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
    from io import BytesIO

    compiled_graph = workflow.compile()
    graph_png = compiled_graph.get_graph().draw_mermaid_png()
    img = mpimg.imread(BytesIO(graph_png), format='PNG')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    with open(output_path, "wb") as f:
        f.write(graph_png)


if os.getenv("DRAW_AGENT_GRAPHS") == "1":
    save_workflow_graph()
# %%
# 问题示例
# (1) 从郑州出发，坐高铁到安阳怎么走？
# (2) 兰州 车程大约2小时到哪里
# (3) 郑州     出发能去哪里
# rag_message_list=[]
# async def chat_with_agent():
#     print("rag_agent 旅游助手已启动！输入 '退出' 结束对话")
#     print("=" * 50)
#     while True:
#         try:
#             user_input = input("\n请输入您的问题: ").strip()
#             if user_input.lower() in ['退出', 'exit', 'quit']:
#                 print("感谢使用，再见！")
#                 break
#             if not user_input:
#                 continue
#             print("\n正在处理...")
#             rag_message_list.extend([{"role": "user", "content": user_input}])
#             result = await rag_agent.ainvoke({
#                 "messages": rag_message_list,
#                 "generate_times": 0
#             })
#             response = result.get("messages")[-1].content
#             print(f"\n助手回答: {response}")
#             rag_message_list.extend([{"role":"assistant","content": response}])
#         except KeyboardInterrupt:
#             print("\n\n对话已中断")
#             break
#         except Exception as e:
#             print(f"\n发生错误: {str(e)}")
#
#
# if __name__ == "__main__":
#     asyncio.run(chat_with_agent())
