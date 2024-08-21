from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import TextLoader, PyPDFLoader

class ChatPDF:
    
    vector_store = None
    retriever = None
    chain = None
    
    def __init__(self):
        self.model = ChatOllama(model="Llama3.1")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s>[INST]Eres un asistente para tareas de responder preguntas. Utiliza las siguientes piezas de contexto 
            recuperado para responder la pregunta. Si no sabes la respuesta, simplemente di que no la sabes. 
            Usa un máximo de tres oraciones y mantén la respuesta concisa.[/INST]</s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]                            
            """
        )

    def ingest(self, file_path_or_url: str):
        if  file_path_or_url.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path=file_path_or_url)
        else:
            loader = TextLoader(file_path=file_path_or_url, encoding="utf-8") 
        documents = loader.load()
        for document in documents:
            print("Document Source:", document.metadata.get('source', 'Unknown'))
            print("Page content:", document.page_content)
            print()
        text_chunks = []
        for document in documents:
            text_chunks.extend(self.text_splitter.split_text(document.page_content))
    
        chunks = [Document(page_content=chunk, metadata=document.metadata) for chunk in text_chunks]
    
        chunks = filter_complex_metadata(chunks)
    
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
    
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                    | self.prompt
                    | self.model
                    | StrOutputParser())
        
    def ask(self, query: str):
        if not self.chain:
            return "Por favor, primero seleccione algun documento txt o pdf"

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
