from __future__ import annotations
from pydoc import Doc
from typing import List, Iterable
from pathlib import Path
from langchain_core.documents import Documents
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exceptions.exception import DocumentPortalException
from fastapi import UploadFile


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

def load_documents(paths:Iterable[Path])->List[Documents]:
    """Load docs using appropriate loader based on extension."""

    docs: List[Documents] = []
    try:
        for p in paths:
            ext = p.suffix().lower()
            if ext == '.pdf':
                loader = PyPDFLoader(str(p))
            elif ext == '.docx':
                loader = Docx2txtLoader(str(p))
            elif ext == '.txt': 
                loader = TextLoader(str(p), encoding='utf-8')
            else:
                log.warning("Unsuported extension skipped", path=str(p))
                continue

            docs.extend(loader.load())
            return docs
        
        log.info('Document loaded', count=len(docs))

    except Exception as e:
        log.error('Failed to load document', error=str(e))
        raise DocumentPortalException('Error loading documents', e) from e


class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()."""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()
