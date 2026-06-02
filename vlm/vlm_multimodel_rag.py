"""
使用vlm模型进行pdf分析测试代码
1.前置环境安装
pip install "olmocr[gpu]"
olmOCR只支持本地部署，硬件条件GPU显存高于15 GB
sudo apt-get update
sudo apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

2.模型下载
mkdir olmOCR-7B-0725-FP8
modelscope download --model allenai/olmOCR-7B-0725-FP8 --local_dir ./olmOCR-7B-0725-FP8
3.通过vllm启动服务:
vllm serve ./olmOCR-7B-0725-FP8 \
  --served-model-name olmocr \
  --max-model-len 16000 \
  --port 3000

4.测试效果
# Download a sample test PDF
curl -o olmocr-sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf
#调用原生api生成对应的结果
#(1) OCR识别1份文件
先进入文件所在的目录:
cd /root/autodl-tmp/multimodel_RAG/vlm/
python -m olmocr.pipeline ./localworkspace --markdown --model /root/autodl-tmp/olmOCR-7B-0725-FP8   --pdfs ./olmocr-sample.pdf
#(2) OCR识别多份文件
先进入文件所在的目录:
cd /root/autodl-tmp/multimodel_RAG/vlm/
python -m olmocr.pipeline ./localworkspace --markdown --model /root/autodl-tmp/olmOCR-7B-0725-FP8  --pdfs ./*.pdf
"""
#%%
# 下面通过将olmOCR做一些改造，使得其具备VLM模型的能力
"""
通过vllm启动olmocr服务引擎:
vllm serve ./olmOCR-7B-0725-FP8 \
  --served-model-name olmocr \
  --max-model-len 16000 \
  --port 3000

多模态olmOCR处理链路：
将多页pdf文档转成多页png图片->

"""
import base64
import requests
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASETS_DIR, VLM_RESULT_MARKDOWN_DIR, get_env, load_project_env, require_env

load_project_env()
VLLM_ENDPOINT = get_env("VLLM_ENDPOINT", "http://localhost:3000/v1/chat/completions")
MODEL_NAME = "olmocr"  # 必须与 vLLM 的 --served-model-name 一致
#将单个pdf_file转成若干张image存放到指定路径下
def get_page_images(input_pdf_file,output_image_path,MAX_PAGES=200):
    #将PDF转成images，dpi为精度，显存 vs 精度一般取值200~300够用
    print(f"当前正在处理文件:{input_pdf_file}")
    pages = convert_from_path(input_pdf_file, dpi=200)
    print("pages:",pages)
    #存放每个pdf页面转成图片后的数组
    page_images_list = []
    for i, img in enumerate(pages[:MAX_PAGES], start=1):
        # 将较大的图片压缩到1600*1600像素以内
        max_side = max(img.size)
        if max_side > 1600:
            scale = 1600 / max_side
            img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
        file_name = file.split("/")[-1].replace(".pdf", "")
        if '香港' in file_name:
            try:
                os.mkdir(fix_url + "xianggang")
            except:
                pass
            file_fold_name = "xianggang/"
        elif '澳门' in file_name:
            try:
                os.mkdir(fix_url + "aomen")
            except:
                pass
            file_fold_name = 'aomen/'
        elif '台北' in file_name:
            try:
                os.mkdir(fix_url + "taiwan")
            except:
                pass
            file_fold_name = 'taiwan/'
        else:
            os.mkdir(fix_url + file_name)
            file_fold_name = file_name
        one_pdf_image_output_path=output_image_path+file_fold_name+f"__page_{i}.png"
        print("one_pdf_image_output_path:",one_pdf_image_output_path)
        img.save(one_pdf_image_output_path, "PNG")
        page_images_list.append(one_pdf_image_output_path)
    print("page_images_list:",page_images_list)
    return page_images_list

# input_pdf_file = "/root/autodl-tmp/multimodel_RAG/vlm/我是驴友-台北旅游攻略.pdf"