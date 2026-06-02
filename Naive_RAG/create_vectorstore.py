"""
本文件用于创建FAISS向量数据库旅游景点攻略
"""
#%%
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.tools.retriever import create_retriever_tool
from pathlib import Path
import sys
import pandas as pd
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASETS_DIR, FAISS_INDEX_DIR, load_project_env, require_env

load_project_env()
#处理csv数据
import pandas as pd
travel_dataframe=pd.read_csv(DATASETS_DIR / "travel_guide.csv",encoding="gbk")
print(travel_dataframe)
#将文档做数据处理,逐列抽取、拼装数据
#%%
print(travel_dataframe.columns)
# Index(['目的地', '交通安排', '住宿推荐', '必打卡景点', '美食推荐', '实用小贴士', '旅行感悟'], dtype='object')
#%%
destinations = travel_dataframe['目的地'].tolist()
transportations = travel_dataframe['交通安排'].tolist()
accommodations = travel_dataframe['住宿推荐'].tolist()
attractions = travel_dataframe['必打卡景点'].tolist()
foods = travel_dataframe['美食推荐'].tolist()
tips = travel_dataframe['实用小贴士'].tolist()
thoughts = travel_dataframe['旅行感悟'].tolist()
#存放最终的数据集
final_destinations=destinations
final_transportations=[]
final_accommodations=[]
final_attractions=[]
final_foods=[]
final_tips=[]
final_thoughts=[]

#将数据集做一下变换
for dest,trans,accom,attr,food,tip,thoug in zip(destinations,transportations,accommodations,attractions,foods,tips,thoughts):
    final_transportations.append(dest+"交通路线:"+str(trans))
    final_accommodations.append(dest+"推荐住宿:"+str(accom))
    final_attractions.append(dest+"必打卡景点:"+str(attr))
    final_foods.append(dest+"推荐美食"+str(food))
    final_tips.append(dest+"注意事项:"+str(tip))
    final_thoughts.append(dest+"旅行感悟:"+str(thoug))

#将数据集拼装成向量数据库可以接收的列表形式
whole_data_list=final_transportations+final_accommodations+final_attractions+final_foods+final_tips+final_thoughts
print(whole_data_list[0:10])

#%%
print(len(whole_data_list)) #4824条
#%%
# ---------------------------------------------------------------------------
# 将列表数据包装成Document对象
def create_documents_from_list(data_list, metadata_field="content_type"):
    """将字符串列表转换为Document对象列表"""
    documents = []
    for i, text in enumerate(data_list):
        # 创建元数据
        metadata = {
            "source": "travel_guide.csv",
            "item_id": i
        }
        # 创建Document对象
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
    return documents


# 创建Document对象列表
documents_list = create_documents_from_list(whole_data_list)
# documents_list=batch_create_faiss_vectorstore(source_dir="/root/autodl-tmp/multimodel_RAG/", save_path="./faiss_traval_db")
print(documents_list)
#%%
# 初始化嵌入模型
from langchain_community.embeddings import DashScopeEmbeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=require_env("QWEN_API_KEY")
)
#%%
# 创建向量库并保存到本地
# 创建向量数据库
def create_vector_store_from_documents(documents, embeddings, save_path=FAISS_INDEX_DIR):
    """从Document列表创建向量数据库并保存"""
    # 如果文档很大，可以分批处理
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    # 保存到本地
    vector_store.save_local(str(save_path))
    print(f"向量库已保存到: {save_path}")
    print(f"索引中文档数量: {vector_store.index.ntotal}")
    return vector_store

# 创建并保存向量数据库
vector_store = create_vector_store_from_documents(documents_list, embeddings)
print(vector_store)
#%%
#从本地加载向量数据库
def load_faiss_vectorstore(load_path):
    # 加载向量库
    vector_store = FAISS.load_local(
        folder_path=load_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,  # 处理潜在危险的序列化/反序列化操作
    )
    print(f"向量库已从 {load_path} 加载")
    return vector_store

vector_store=load_faiss_vectorstore(load_path=str(FAISS_INDEX_DIR))
print("向量数据库加载成功:",vector_store)
#%%
rag_retriever=vector_store.as_retriever(search_type="similarity_score_threshold",
                                    search_kwargs={"score_threshold":0.6,"k": 1})
result1=rag_retriever.invoke("黄姚古镇旅游线路推荐一下")
print("result1:",result1)
#%%
rag_retriever=vector_store.as_retriever(search_type="similarity_score_threshold",
                                    search_kwargs={"score_threshold":0.6,"k": 1})
result2=rag_retriever.invoke("去殷墟景区怎么走")
print("result2:",result2)
#%%
#再来看一个检索不到的案例
result3=rag_retriever.invoke("从上海出发有哪些景点可以去？")
print("result3:",result3)


