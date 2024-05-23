from typing import List, Optional, Iterable
import copy
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class CustomRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):

    def create_documents(
            self, texts: List[str], metadatas: Optional[List[dict]] = None, pdf_file: str = ""
    ) -> List[Document]:
        """Create documents from a list of texts, each prefixed with the file name."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        # Generate file descriptor, remove .pdf suffix and add descriptor tag
        file_desc = f"[desc]{pdf_file[:-4]}[desc] "
        for i, text in enumerate(texts):
            index = 0
            previous_chunk_len = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    offset = index + previous_chunk_len - self._chunk_overlap
                    index = text.find(chunk, max(0, offset))
                    metadata["start_index"] = index
                new_doc = Document(page_content=file_desc + chunk, metadata=metadata)
                documents.append(new_doc)
                previous_chunk_len = len(chunk)
        return documents

    def split_documents(self, documents: Iterable[Document], pdf_file: str) -> List[Document]:
        """Split documents and prepend each page content with the file name."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_documents(texts, metadatas, pdf_file=pdf_file)