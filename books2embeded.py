import os
import pandas as pd
import torch
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from custom_recursive_text_splitter import CustomRecursiveCharacterTextSplitter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_SPLITTER = CustomRecursiveCharacterTextSplitter(chunk_size=230, chunk_overlap=30)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

pdfs_liver_dir = 'guide_books/pdfs_liver'
textbooks_all_dir = 'guide_books/textbooks_all'
wiki_liver_dir = 'guide_books/wiki_liver'

output_base_dir = 'sections'
subfolders = ['pdfs_liver', 'textbooks_all', 'wiki_liver']
for folder in subfolders:
    output_dir = os.path.join(output_base_dir, folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


# 将pdf文件、txt文件进行分块
def process_files(file_dir, file_type='pdf', output_subfolder=''):
    documents = []
    files = [f for f in os.listdir(file_dir) if f.endswith('.' + file_type)]
    for file in tqdm(files):
        file_path = os.path.join(file_dir, file)
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
            text = loader.load()
        else:
            loader = TextLoader(file_path, encoding='utf-8')
            text = loader.load()
        chunks = TEXT_SPLITTER.split_documents(text, file)
        print(f"Processed {file_type} file: {file}, Chunks: {len(chunks)}")
        df = pd.DataFrame(chunks, columns=['page_content', 'metadata', 'dtype'])
        output_file = os.path.join(output_base_dir, output_subfolder, file.replace('.' + file_type, '.csv'))
        df.to_csv(output_file, index=False, encoding='utf-8', escapechar='\\')
        documents.extend(chunks)
    print(f"{pdfs_liver_dir}: {len(documents)}")
    return documents


liver_docs = process_files(pdfs_liver_dir, 'pdf', 'pdfs_liver')
print(len(liver_docs))
liver_docs += process_files(wiki_liver_dir, 'txt', 'wiki_liver')
print(len(liver_docs))
# 将liver的pdf和txt嵌入成一个向量库
chroma_liver = Chroma.from_documents(documents=liver_docs, embedding=embeddings, persist_directory='./chromadb/chroma_liver')


all_docs = liver_docs.copy()
print(len(all_docs))
all_docs += process_files(textbooks_all_dir, 'txt', 'textbooks_all')
print(len(all_docs))
# 将liver和MedQA的所有文档都嵌入成一个向量库
chroma_all = Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory='./chromadb/chroma_all')

print("Combined database successfully created.")

