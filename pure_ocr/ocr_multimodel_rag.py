"""
多模态pdf知识检索引擎项目前置步骤说明:
第一步:linux系统中pdf2img需要安装工具poppler-utils
sudo apt-get update
sudo apt-get install poppler-utils

第二步: 安装ocr解析引擎tesseract
sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn
which tesseract 找到对应的路径

第三步:手动下载模型,去modelscope上搜索如下模型进行下载，推荐下载到数据盘:
- transformer-structure-recognition(表格识别模型)
- resnet18.a1_in1k(图像分类模型)
- yolo_x_layout(目标检测模型)
下载命令
modelscope download --model microsoft/table-transformer-structure-recognition  --local_dir /root/autodl-tmp
modelscope download --model timm/resnet18.a1_in1k  --local_dir /root/autodl-tmp
modelscope download --model AI-ModelScope/yolo_x_layout --local_dir /root/autodl-tmp

第四步:修改源码：
vi /root/miniconda3/lib/python3.12/site-packages/timm/models/_builder.py 159行增加代码： load_from="file""/root/autodl-tmp/resnet18.al_inlk/model.safetensors'

"""
#%%
from __future__ import annotations
import os
import asyncio
from typing import Literal
from dotenv import load_dotenv
load_dotenv('/root/autodl-tmp/multimodel_RAG/.env')
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#%%
#2. 知识库检索数据集准备
import os
# 设置环境变量强制使用本地模型,首次需要从HF_mirror上下载
# 有了依赖库之后，我们就可以使用 UnstructuredLoader 来解析 PDF 文档了，对于给定的文档，我们可以按照如下方式进行解析：
# 这一步下载可能比较慢
import pytesseract #ocr的时候会调用pytesseract库,因此需要安装并指定tesseract_cmd的程序启动路径为你的安装路径,
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract' #which tesseract 查看安装路径
#解析表格的模型：默认模型是table-transformer-structure-recognition，并通过load_agent()函数来初始化加载本地默认路径下的已经下载好的模型
from unstructured_inference.models import tables
from unstructured_inference.logger import logger
import os
# 将yolo模型的本地模型路径修改进这个配置文件中
os.environ["UNSTRUCTURED_DEFAULT_MODEL_INITIALIZE_PARAMS_JSON_PATH"] = "/root/autodl-tmp/multimodel_RAG/model_init_parameters.json"
def custom_load_table_model():
    """Loads the Table agent."""
    if getattr(tables_agent, "model", None) is None:
        with tables_agent._lock:
            if getattr(tables_agent, "model", None) is None:
                logger.info("Loading the Table agent ...")
                tables_agent.initialize("/root/autodl-tmp/table-transformer-structure-recognition/")
    return
tables_agent = tables.tables_agent
# 覆盖默认的表格模型加载函数
tables.load_agent = lambda: custom_load_table_model()
pdf_path = "/root/autodl-tmp/multimodel_RAG/datasets/gang_ao_pdf/我是驴友-台北旅游攻略.pdf" #提取所有的pdf文件
output_dir = "/root/autodl-tmp/multimodel_RAG/pure_ocr/image_path" #图片输出路径
os.makedirs(output_dir, exist_ok=True)
#%%
#第一步：提取文字信息
from unstructured.partition.pdf import partition_pdf
extract_elements = partition_pdf(
    filename=pdf_path,
    infer_table_structure=True,   # 支持表格结构检测
    strategy="hi_res",            # ocr+表格识别模式
    languages=["chi_sim","eng"],  # ocr tesseract 的language为中英文混合识别
)
#%%
print(extract_elements)
#%%
import fitz
# 第二步: 提取图片信息并保存到指定目录(output_dir)下
# doc = fitz.open(pdf_path) #使用PyMuPDF库来打开和加载 PDF 文件到内存中
image_folder = {}  #{"page_no": [存放某个page中的所有图片路径列表]}
def extract_images_from_pdf(pdf_path, output_dir):
    """
    PDF文档提取各类元素函数，将图片保存到相应路径中
    """
    doc = fitz.open(pdf_path)#使用PyMuPDF库来打开和加载 PDF 文件到内存中
    image_folder = {}  #{"page_no": [存放某个page中的所有图片路径]}
    os.makedirs(output_dir, exist_ok=True)
    for page_num, page in enumerate(doc, start=1):
        image_folder[page_num] = []
        for img_index, img_info in enumerate(page.get_images(full=True), start=1):
            xref = img_info[0]
            pix = fitz.Pixmap(doc, xref)
            # 处理色彩空间
            # pix.n 表示 颜色空间中的分量数量（number of color components），具体对应关系：
            # pix.n = 1：灰度图像（Gray），只有1个颜色通道
            # pix.n = 3：RGB图像，有3个颜色通道（红、绿、蓝）
            # pix.n = 4：RGBA图像，有4个通道（红、绿、蓝、透明度）
            # pix.n = 5：CMYK图像（特殊情况），有4个颜色通道（青、品红、黄、黑）
            if pix.n >= 5:  # CMYK或其他非标准色彩空间
                pix = fitz.Pixmap(fitz.csRGB, pix)
            # 保存图像
            img_filename = f"page{page_num}_img{img_index}.png"
            img_path = os.path.join(output_dir, img_filename)
            pix.save(img_path)
            image_folder[page_num].append(img_path)
    doc.close()
    return image_folder
image_folder=extract_images_from_pdf(pdf_path, output_dir)
#%%
print(image_folder)
#%%
# 第三步：将pdf转换为Markdown文档输出至指定路径下:
from pathlib import Path
from typing import List, Set, Dict, Optional
import logging
from html2text import html2text
logging.basicConfig(level=logging.INFO)
"""Markdown文档转换器
将提取的元素（标题、表格、图片等）转换为Markdown格式文档
Attributes:
    output_dir: 输出目录路径
    output_md_file: 输出Markdown文件名
    logger: 日志记录器
"""
class MarkdownConverter:
    def __init__(self, output_dir: str = ".", output_md_file: str = "ocr_output.md"):
        """初始化转换器
        Args:
            output_dir: 输出目录路径，默认为当前目录
            output_md_file: 输出Markdown文件名，默认为output.md
        """
        self.output_dir = Path(output_dir)
        self.output_md_file = self.output_dir / output_md_file
        self.logger = logging.getLogger(__name__)

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    def convert_elements_to_markdown(
            self,
            extract_elements: List,
            image_folder: Dict[int, List[str]]
    ) -> None:
        """将提取的元素转换为Markdown格式并保存到文件
        Args:
            extract_elements: 从文档中提取的元素列表，每个元素应包含category、text和metadata属性
            image_folder: 按页码组织的图片文件路径字典，格式为 {page_num: [img_path1, img_path2, ...]}
        Returns:
            None: 结果将保存到指定的Markdown文件中
        """
        # 初始化Markdown行列表
        markdown_lines: List[str] = []
        # 用于记录已经插入过的图片路径，避免重复插入
        inserted_images: Set[str] = set()
        self.logger.info(f"开始转换 {len(extract_elements)} 个元素到Markdown格式")
        # 遍历所有提取的元素
        for element in extract_elements:
            category = element.category
            text = element.text
            page_num = element.metadata.page_number
            try:
                # 根据元素类别进行不同的处理
                if category == "Title":
                    self._process_title_element(text, markdown_lines)
                elif category in ["Header", "Subheader"]:
                    self._process_header_element(text, markdown_lines)
                elif category == "Table":
                    self._process_table_element(element, markdown_lines)
                elif category == "Image":
                    self._process_image_element(
                        page_num, image_folder, inserted_images, markdown_lines
                    )
                else:
                    # 对于其他类型的元素（如正文文本），直接添加文本
                    self._process_text_element(text, markdown_lines)
            except Exception as e:
                self.logger.error(f"处理元素时发生错误 (类别: {category}, 页码: {page_num}): {str(e)}")
                # 错误处理：将原始文本添加到输出中，避免中断整个流程
                markdown_lines.append(text + "\n")
        # 将生成的Markdown内容写入文件
        self._write_markdown_file(markdown_lines)
        self.logger.info(f"转换完成，已生成 {self.output_md_file} 和对应的图片文件夹")

    def _process_title_element(self, text: str, markdown_lines: List[str]) -> None:
        """处理标题元素
        标题元素以"# "开头，但如果已经以"- "开头（如列表项），则保持原样
        Args:
            text: 标题文本
            markdown_lines: 用于存储Markdown行的列表
        """
        stripped_text = text.strip()
        if stripped_text.startswith("- "):
            # 如果标题以列表项开头，保持原样（可能是一个列表标题）
            markdown_lines.append(text + "\n")
        else:
            # 否则作为一级标题
            markdown_lines.append(f"# {text}\n")
    def _process_header_element(self, text: str, markdown_lines: List[str]) -> None:
        """处理页眉/副标题元素
        页眉和副标题都转换为二级标题
        Args:
            text: 标题文本
            markdown_lines: 用于存储Markdown行的列表
        """
        markdown_lines.append(f"## {text}\n")


    def _process_table_element(self, element, markdown_lines: List[str]) -> None:
        """处理表格元素
        尝试将HTML表格转换为Markdown格式，如果不可用则回退到纯文本
        Args:
            element: 包含表格数据的元素对象
            markdown_lines: 用于存储Markdown行的列表
        """
        # 检查元素是否包含HTML格式的表格数据
        if hasattr(element.metadata, "text_as_html") and element.metadata.text_as_html:
            try:
                # 将HTML表格转换为Markdown格式
                table_markdown = html2text(element.metadata.text_as_html)
                markdown_lines.append(table_markdown + "\n")
            except Exception as e:
                self.logger.warning(f"HTML表格转换失败，使用纯文本回退: {str(e)}")
                markdown_lines.append(element.text + "\n")
        else:
            # 如果没有HTML格式，直接使用纯文本
            markdown_lines.append(element.text + "\n")
    def _process_image_element(
            self,
            page_num: int,
            image_folder: Dict[int, List[str]],
            inserted_images: Set[str],
            markdown_lines: List[str]
    ) -> None:
        """处理图片元素
        为当前页码查找对应的图片文件，并插入Markdown图片引用
        使用集合避免重复插入同一图片
        Args:
            page_num: 当前页码
            image_folder: 按页码组织的图片文件路径字典
            inserted_images: 已插入图片的集合
            markdown_lines: 用于存储Markdown行的列表
        """
        # 获取当前页码对应的所有图片路径
        image_paths = image_folder.get(page_num, [])

        if not image_paths:
            self.logger.debug(f"页码 {page_num} 未找到对应的图片文件")
            return

        for img_path in image_paths:
            # 检查图片是否已经插入过
            if img_path not in inserted_images:
                # 构建Markdown图片引用
                # 假设图片文件与Markdown文件在同一目录或子目录中
                markdown_lines.append(f"![Image]({img_path})\n")

                # 记录已插入的图片
                inserted_images.add(img_path)
                self.logger.debug(f"插入图片: {img_path}")
            else:
                self.logger.debug(f"跳过已插入的图片: {img_path}")
    def _process_text_element(self, text: str, markdown_lines: List[str]) -> None:
        """处理普通文本元素
        对于非特殊类型的元素，直接添加文本内容
        Args:
            text: 文本内容
            markdown_lines: 用于存储Markdown行的列表
        """
        markdown_lines.append(text + "\n")
    def _write_markdown_file(self, markdown_lines: List[str]) -> None:
        """将Markdown内容写入文件
        Args:
            markdown_lines: 包含所有Markdown行的列表
        Raises:
            IOError: 当文件写入失败时抛出
        """
        try:
            # 将所有Markdown行连接成字符串
            markdown_content = "".join(markdown_lines)
            # 写入文件，使用UTF-8编码确保中文字符正确处理
            with open(self.output_md_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            self.logger.info(f"Markdown文件已保存: {self.output_md_file}")
        except IOError as e:
            self.logger.error(f"写入Markdown文件失败: {str(e)}")
            raise


#创建转换器实例化后调用convert_elements_to_markdown得到md结果
converter = MarkdownConverter(
    output_dir="/root/autodl-tmp/multimodel_RAG/pure_ocr/result_markdown",
    output_md_file="converted_document.md"
)
converter.convert_elements_to_markdown(extract_elements, image_folder)
#%%
#附加：元素检测可视化,进行文档区域布局暂时
from langchain_unstructured import UnstructuredLoader
elements_layout_outputs = UnstructuredLoader(
    file_path=pdf_path,
    strategy="hi_res",  # 高分辨率分析策略
    infer_table_structure=True,  # 启用表格结构识别
    languages=["chi_sim","eng"],  # 支持中文和英文OCR
)
#%%
# 元素检测分析列表
doc_lists = []
for document_element in elements_layout_outputs.lazy_load():
    doc_lists.append(document_element)
print("doc_lists:",doc_lists)
#%%
#把PDF页面及元素渲染成图片做可视化，绘制出类别定位框（标题、表格、图片、文本等）。
import fitz
# pip install matplotlib==3.7.1
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
def pdf_document_element_layout_detect(pdf_page, detected_segments):
    """
    在PDF页面图像上可视化文档元素布局检测结果
    参数:
        pdf_page: PyMuPDF页面对象
        detected_segments: 包含检测到的文档元素及其坐标信息的列表
    """
    # 将PDF页面渲染为像素图
    pixmap_pages = pdf_page.get_pixmap()
    print("页面像素图信息:", pixmap_pages)
    #，将 PyMuPDF 的图像数据转换为 PIL 图像
    pil_converted_image = Image.frombytes(
        mode="RGB",
        size=[pixmap_pages.width, pixmap_pages.height],
        data=pixmap_pages.samples
    )
    # 创建可视化图形
    visualization_figure, visualization_axes = plt.subplots(
        nrows=1,
        figsize=(12, 12)
    )
    visualization_axes.imshow(pil_converted_image)
    # 定义文档元素类别与对应颜色映射
    ELEMENT_CATEGORY_COLORS = {
        "Title": "#6200EA",  # 深紫色 - 标题
        "Image": "#03DAC6",  # 青绿色 - 图像
        "Table": "#FF9800",  # 橙色   - 表格
    }
    DEFAULT_BOUNDING_COLOR = "#111827"  # 默认颜色（文本元素）
    # 收集所有出现的元素类别
    observed_categories = set()
    # 遍历所有检测到的文档元素
    for element_data in detected_segments:
        # 提取元素坐标信息
        element_points = element_data["coordinates"]["points"]
        reference_width = element_data["coordinates"]["layout_width"]
        reference_height = element_data["coordinates"]["layout_height"]
        print(
            f"元素坐标点: {element_points}, "
            f"参考宽度: {reference_width}, "
            f"参考高度: {reference_height}"
        )
        # 将相对坐标转换为绝对坐标（适配实际图像尺寸）
        absolute_coordinates = [
            (coord_x * pixmap_pages.width / reference_width,
                coord_y * pixmap_pages.height / reference_height
            )
            for coord_x, coord_y in element_points
        ]
        # 确定当前元素类别的边框颜色
        element_category = element_data["category"]
        observed_categories.add(element_category)

        bounding_box_color = ELEMENT_CATEGORY_COLORS.get(
            element_category,
            DEFAULT_BOUNDING_COLOR
        )
        # 在图像上绘制元素边界框
        bounding_polygon = patches.Polygon(
            absolute_coordinates,
            linewidth=1,
            edgecolor=bounding_box_color,
            facecolor="none"
        )
        visualization_axes.add_patch(bounding_polygon)
    # 创建图例
    legend_elements = [
        patches.Patch(color=DEFAULT_BOUNDING_COLOR, label="Text")
    ]
    for category_type in ["Title", "Image", "Table"]:
        if category_type in observed_categories:
            legend_elements.append(
                patches.Patch(
                    color=ELEMENT_CATEGORY_COLORS[category_type],
                    label=category_type
                )
            )
    # 配置图形显示
    visualization_axes.axis("off")
    visualization_axes.legend(
        handles=legend_elements,
        loc="upper right"
    )
    plt.tight_layout()
    plt.show()

def show_page_layout_result(file_path,doc_lists: list, page_no: int) -> None:
    #- 打开 PDF，定位到第 page_number 页。
    pdf_page = fitz.open(file_path).load_page(page_no - 1)
    #提取文档
    page_docs = [doc for doc in doc_lists if doc.metadata.get("page_number") == page_no]
    print(page_docs)
    #提取坐标类别
    segments = [doc.metadata for doc in page_docs]
    print(segments)
    #将坐标、类别绘制在原pdf文档上
    pdf_document_element_layout_detect(pdf_page, segments)
    for doc in page_docs:
        print(f"{doc.page_content}\n")
#%%
# 此时我们就能看到每一个PDF页面里面提取的元素了：
show_page_layout_result(pdf_path,doc_lists, 3)
#%%
load_dotenv('/root/autodl-tmp/multimodel_RAG/.env')
# 然后，通过markdown文档来创建知识检索引擎：
#%%
from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(
        api_key=os.getenv("QWEN_API_KEY",""),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v3",
        check_embedding_ctx_length = False,
)
print(embeddings_model)
#%%
file_path = "/root/autodl-tmp/multimodel_RAG/pure_ocr/result_markdown/converted_document.md"
with open(file_path, "r", encoding="utf-8") as f:
    md_content = f.read()

from langchain_text_splitters import MarkdownHeaderTextSplitter
"""
headers_to_split_on:第一个元素：Markdown 标题标记符（#, ##, ### 等）
第二个元素：该层级标题在分割后的元数据中的名称

MarkdownHeaderTextSplitter:它会查找文档中的所有标题
只关注 headers_to_split_on 中指定的标题层级
在每个指定的标题处进行文档分割
未指定的标题层级将被忽略（不会触发分割）
"""
headers_to_split_on = [("#", "Header 1"),("##", "Header 2")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md_content)
#%%
print(md_header_splits[3])
#%%
# 指定向量嵌入的批次大小，qwen-embedding的批次数不要超过20
batch_size = 10
# 创建一个批次文档的初始向量库确保不超过列表范围
initial_docs = md_header_splits[:min(batch_size, len(md_header_splits))]
print(len(initial_docs))
vector_store = FAISS.from_documents(initial_docs, embeddings_model)
#%%
# 将剩余的文档分批添加进去
for i in range(batch_size, len(md_header_splits), batch_size):
    batch = md_header_splits[i:i + batch_size]
    vector_store.add_documents(batch)

# 保存向量库
vector_store.save_local("gang_ao_tai")
print("向量库构建成功")
#%%
#加载持久化存储的向量库，进行检索
import os
from dotenv import load_dotenv
load_dotenv()
embeddings_model = OpenAIEmbeddings(
        api_key=os.getenv("QWEN_API_KEY",""),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v3",
        check_embedding_ctx_length = False,
)
multi_model_material_path =  "gang_ao_tai"
vector_store = FAISS.load_local(
    folder_path=multi_model_material_path,
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True,
)
#%%
multi_model_retriever=vector_store.as_retriever(search_type="similarity", #_score_threshold
                                    search_kwargs={"k": 5}) #"score_threshold":0.3,
search_result=multi_model_retriever.invoke("去台湾玩有哪些方式？")
print(search_result)
#%%
search_result=multi_model_retriever.invoke("大陆地区人民来台其停留期间不得超过几天？")
print(search_result)
#%%
search_result=multi_model_retriever.invoke("从大陆往台湾拨号电话是多少？")
print(search_result)
#%%
#会搜索不到,思考为什么？表格信息并没有得到很好的检索
search_result=multi_model_retriever.invoke("夫妻同游需要提供什么")
print(search_result)