import fitz
import os


# 去除pdf中参考文献等等检索无关的信息
def remove_last_n_pages(pdf_path, n):
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)

    # 创建一个新的PDF文档
    new_pdf_document = fitz.open()

    # 将保留的页数复制到新的PDF文档
    for page_num in range(n):
        new_pdf_document.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)

    output_dir = "processed_pdfs"
    os.makedirs(output_dir, exist_ok=True)

    new_pdf_filename = os.path.basename(pdf_path)
    new_pdf_path = os.path.join(output_dir, new_pdf_filename)
    new_pdf_document.save(new_pdf_path)

    pdf_document.close()
    new_pdf_document.close()

    return new_pdf_path


pdf_path = "pdfs/中国药物性肝损伤诊治指南（2023年版）.pdf"  # 替换为PDF文件路径
n = 18  # 要保留的页数

# 保留PDF文件的前n页
trimmed_pdf_path = remove_last_n_pages(pdf_path, n)
