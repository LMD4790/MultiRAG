"""
三、搭建基于多模态MarkDown文档的Agentic RAG检索引擎
#在跑通了多模态文档转化之后，接下来我们基于转化后的多模态MarkDown文档来创建一个Agentic RAG引擎。项目完整代码如下：

- 安装前端框架Agent Chat UI
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui
pnpm install #然后安装前端依赖：
安装LangGraph项目部署工具：
pip install -U "langgraph-cli[inmem]"
"""
from __future__ import annotations
import os
import asyncio
import sys
from pathlib import Path
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing import Literal
from typing_extensions import TypedDict,Annotated    # 如果需要向后兼容
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, END, add_messages
#%%
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_ROOT, VLM_RESULT_MARKDOWN_DIR, load_project_env, require_env, require_path

load_project_env()
os.environ["DEEPSEEK_API_KEY"] = require_env("DEEPSEEK_API_KEY")
# print(os.getenv("DEEPSEEK_API_KEY"))
#%%
MODEL_NAME = "deepseek-chat"
model = init_chat_model(model=MODEL_NAME, model_provider="deepseek", temperature=0)
# - grader_model：用于判断文档相关性的小助手模型。
grader_model = init_chat_model(model=MODEL_NAME, model_provider="deepseek", temperature=0)
print(grader_model)
embeddings_model = OpenAIEmbeddings(
        api_key=require_env("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v3",
        check_embedding_ctx_length = False,
)
#%%
#加载多模态解析好的路径下的vectordb
multi_model_material_path = require_path(VLM_RESULT_MARKDOWN_DIR, "multimodal FAISS index")
vector_store = FAISS.load_local(
    folder_path=str(multi_model_material_path),
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True,
)
# retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
multi_model_retriever=vector_store.as_retriever(search_type="similarity_score_threshold",
                                    search_kwargs={"score_threshold":0.3,"k": 8})

# print(await multi_model_retriever.ainvoke("台湾旅行行程规划"))
#%%
# 使用langchain内置方法构建多模态向量数据库加载与检索工具构建
gang_ao_tai_retriever_tool = create_retriever_tool(
    multi_model_retriever,
    name="gang_ao_tai_retrieve_tool",
    description="检索并返回香港、澳门、台湾旅游相关的知识工具",
)
#%%
# 指定多模他RAG Agent只能回复港澳台旅游相关的知识
SYSTEM_INSTRUCTION = """你是一个香港、澳门、台湾旅游向导，你善于回答有关港澳台旅游相关的一切问题。
包括：港澳台的旅行规划、景点介绍、出发准备、交通线路、相关攻略与书籍、天气情况、宗教文化、住宿、购物、餐饮、当地方言等。
如果用户的问题与港澳台旅游无关，请回复：我无法回答与港澳台旅游无关的问题。
技能：遇到香港、澳门和台湾旅游的问题，你必须调用gang_ao_tai_retrieve_tool进行知识库检索知识进行回复。
请注意：
- 请勿回答除港澳台旅游之外的问题。
- 若用户本轮问题表述已经完整请勿结合历史会话进行问题融合,仅对本轮问题进行改写;若用户本轮问题表述不完整,需要结合历史问题进行补充式改写
举例1:
原始问题:历史问题:香港旅游\n本轮问题:有什么好玩的地方 
优化后问题:去香港旅游什么好玩的地方推荐
举例2:
原始问题:历史问题:澳门有什么好吃的\n本轮问题:有什么好玩的? 
优化后问题:澳门有什么好玩的景点? 
举例3:
原始问题:历史问题:台北的旅游攻略?\n本轮问题:澳门怎么去比较方便?
优化后问题:澳门的交通线路有哪些?

"""
#评估提示词
GRADE_PROMPT = """你是一位评估检索到的文档与用户问题相关性的评分员。\n
检索到的文档：\n{context}\n\n用户问题：{question}\n"
若相关请返回'yes'，否则返回'no'"""

REFINE_PROMPT="""你的任务是优化或澄清用户的问题，使得修改后的问题能更符合香港、澳门、台湾的旅游相关主题。
注意：港澳台旅游相关指的是例如：旅行规划、景点介绍、出发准备、交通线路、相关攻略与书籍、天气情况、宗教文化、住宿、购物、餐饮、当地方言等。
你只需要输出优化后的用户问题，请勿做多余解释。


------
原始问题:{question}
优化后问题:"""


# - 如果用户问题偏离主题，让模型改写问题，使其更贴近“工具调用 / LangGraph / MCP”等关键概念。
# - 给模型一个问题和上下文，引导它用 Markdown、代码块、图片等方式生成结构化答案。
ANSWER_PROMPT = """您是一位专业的港澳台旅游智能助手。请严格依据为您检索到的'参考文档'，回答用户提出的旅游相关咨询。
### 回答原则
1.答案必须完全依据所提供的上下文信息，不编造、不臆测，确保信息完整且准确。
2.特别对于港澳台旅行规划、景点介绍、出发准备、交通线路、相关攻略与书籍、天气情况、宗教文化、住宿、购物、餐饮、当地方言等内容，请务必使用上下文中的材料。
3.若有相关示例、行程路线、实用代码（如交通APP使用代码）或图片，请直接引用或展示。
- 示例：上下文中的具体行程安排或活动介绍。
- 代码：如签证申请流程模拟代码、地铁线路查询代码块。
- 图片：如景区地图、美食或地标图片输出类似： `![图片描述](/img_path/fold/pic.png)`。若高度相关，请将其嵌入回答。
4.使用标准Markdown*格式组织回答，用markdown语法表代码块、表格、文本等。
5. 如果上下文信息无法解答用户问题，请直接回复：“我不知道。”

### 本轮任务：
输入:
用户问题：{question}
参考文档：{context}
输出:"""

#定义状态变量
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    generate_times: int

# LangGraph节点定义
# ---------------------------------------------------------------------------
# - 调用 LLM，根据当前消息决定是否要调用 retriever_tool；
# - 本质上是一个具备工具调用能力的交互节点（如果上下文不足，模型会自动决定调用检索器）。
async def generate_respond(state: State):
    state["generate_times"]+=1 #统计生成次数
    print("进入MultiModelRag_agent,当前生成次数+1:", state["generate_times"])
    response = await model.bind_tools([gang_ao_tai_retriever_tool]).ainvoke(
        [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            *state["messages"],
        ]
    )
    print("generate_respond", response)
    # 追加response到现有消息列表
    state["messages"].append(response)

    #更新state节点信息
    return {"generate_times":state["generate_times"], "messages": state["messages"]}

#%%  
class GradeDoc(BaseModel):
    relevant_score: str = Field(description="如果是用户问题的答案输出:'yes',如果没有检索文档或与用户问题无关或没有解决用户问题输出:'no'.")
# - 使用 grader_model 判断：检索到的文档是否与提问有关；如果相关则路由到generate_final_answer节点，否则路由到refine_query节点
async def grade_search_docs(state: State) -> Literal["generate_final_answer", "refine_query"]:
    question=None
    for message in state["messages"][::-1]:
        if isinstance(message, HumanMessage):
            question=message.content
            print(f"这是最近一次用户消息:{question}")
            break
    context = state["messages"][-1].content  # multi_model retriever结果
    if state['generate_times'] > 2:
        print("进入multi_model_rag_agent打分节点,当前生成次数超过最大次数2次，不进行改写自由生成")
        return 'generate_final_answer'
    print(f"进入multimodel_agent打分节点,当前生成次数{state['generate_times']}")
    prompt = GRADE_PROMPT.format(question=question, context=context)
    result = await grader_model.with_structured_output(GradeDoc).ainvoke([
        {"role": "user", "content": prompt}
    ])
    print("评估相关性结果:", result)
    return "generate_final_answer" if result.relevant_score.lower().startswith("y") else "refine_query"

# 优化query使得其更好的回复港澳台旅游相关的问题
async def refine_query(state: State):
    question=None
    for message in state["messages"][::-1]:
        if isinstance(message, HumanMessage):
            question=message.content
            print(f"这是最近一次用户消息:{question}")
            break
    if not question:
        return {"generate_times": state["generate_times"], "messages": [{"role": "user", "content": "改写出错"}]}
    print("进入multi_model_rag agent重写节点:", state)
    prompt = REFINE_PROMPT.format(question=question)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    print("resp:",resp)
    return {"generate_times":state["generate_times"],"messages": [{"role": "user", "content": resp.content}]}

# - 用 LLM + 上下文生成最终答复；
# - 支持代码块与 Markdown 格式。
async def generate_final_answer(state: State):
    for message in state["messages"][::-1]:
        if isinstance(message, HumanMessage):
            print("这是最近一次用户消息")
            question=message.content
            break
    print("进入multi_model_rag agent生成回答节点:", state)
    print("question:", question)
    context = state["messages"][-1].content
    print("context:", context)
    prompt = ANSWER_PROMPT.format(question=question, context=context)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [resp]}

#%%
#创建流程图
workflow = StateGraph(State)
workflow.add_node("generate_respond", generate_respond)
workflow.add_node("retrieve", ToolNode([gang_ao_tai_retriever_tool]))
workflow.add_node("refine_query", refine_query)
workflow.add_node("generate_final_answer", generate_final_answer)
workflow.add_edge(START, "generate_respond")
workflow.add_edge("generate_respond", "retrieve")
workflow.add_conditional_edges("retrieve", grade_search_docs)
workflow.add_edge("generate_final_answer", END)
workflow.add_edge("refine_query", "generate_respond")
multi_model_agent = workflow.compile(name="multi_model_agent")

# 编译智能体并生成入口
if os.getenv("DRAW_AGENT_GRAPHS") == "1":
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
    from io import BytesIO

    graph_png = multi_model_agent.get_graph().draw_mermaid_png()
    img = mpimg.imread(BytesIO(graph_png), format='PNG')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    with open(PROJECT_ROOT / "multimodel_agenticRAG.png", "wb") as f:
        f.write(graph_png)
#%%
# 台湾旅游需要准备什么
# 香港的景点推荐几个
# 澳门旅游怎么说
# 澳门和香港旅游有什么不同点？
# 进行query_refine但最终没优化出来的:香港和拜登的关系是啥
# 台湾、香港、澳门旅游有什么差异点
# async def chat_with_agent():
#     print("MultimodelRAG 旅游助手已启动！输入 '退出' 结束对话")
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
#             result = await multi_model_agent.ainvoke({
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
