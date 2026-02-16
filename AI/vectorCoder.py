""" AI vector search"""

from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# from nvidia import nvidiaLoader, nvidiaHeader, nvidiaSelect
# from arm import armLoader, armSelect # ArmDataProcessor
# from dashConf import dashConfig as conf, dashStyles as style
# from cpu import loadProcessors
#
# def importProcessors(debug=False):
#     processors = loadProcessors(debug=False)
#     amdDf = processors.selectManufacturer('AMD')
#     amdDf[' index'] = range(1, len(amdDf) + 1)
#     intelDf = processors.selectManufacturer('Intel')
#     intelDf[' index'] = range(1, len(intelDf) + 1)
#     otherDf = processors.selectManufacturer('notintelamd')
#     otherDf[' index'] = range(1, len(otherDf) + 1)
#     nvidiaDf = nvidiaLoader('nvidia', columns=nvidiaSelect)
#     nvidiaDf[' index'] = range(1, len(nvidiaDf) + 1)
#     armDf = armLoader('ARM', columns=armSelect)
#     armDf[' index'] = range(1, len(armDf) + 1)
#     df = processors.fullDf
#     return df

codeDir = '/home/wrichter/Documents/Code/Projects/Python/processors'
dbLocation = "./AI/chromeLangChainCodeDb"

class AICodeLoader:
    def __init__(self, codeDir, ragType="code", model="llama3", dbLocation="./AI/chromeLangChainCodeDb"):
        self.codeDir = codeDir
        self.ragType = ragType
        self.model = model
        self.dbLocation = dbLocation
        self.docs = dict(docs=None, ids=None)
        self.loadCodebase()
        self.splitDocs()

    def loadCodebase(self):
        loader = DirectoryLoader(
            self.codeDir,
            glob="**/*.py", # Targets only .py files
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
        )
        docs = loader.load()
        self.splitDocs(docs)

    def splitDocs(self, docs):
        """Splits documents into smaller, manageable chunks."""
        # RecursiveCharacterTextSplitter is good for maintaining context in code
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language="python",
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(self.docs)
        self.docs['docs'] = chunks


class AIDfLoader:
    """
        Loads a DataFrame as documents for the vector storage.
        Used as a Data import for the RAG vector.
    """
    def __init__(self, df, ragType="dataframe", model="llama3", keys=None, metadata=None,
                 dbLocation = "./AI/chromeLangChainDb2"):
        self.docs = dict(docs=None, ids=None)
        self.ragType = ragType
        self.model = model
        self.dbLocation = dbLocation
        self.dbExists = not os.path.exists(dbLocation)
        self.keys = keys if keys else ['manufacturer', 'processor_family', 'microarchitecture', 'code_name', 'model']
        self.metadata = metadata if metadata else {'cores': 'hw_ncores', 'threadspercore': 'hw_nthreadspercore',
                                          'date': 'created_at'}
        if self.dbExists:
            self.loadDocuments(df)

    def loadDocuments(self, df):
        """Loads a DataFrame as  if the vector store does not exist."""
        documents = []
        ids = []
        for i, row in df.iterrows():
            # print(i)
            content = ""
            for key in self.keys:
                content += str(row[key]) + " "
            metadata = {entry: str(row[self.metadata[entry]]) for entry in self.metadata}
            document = Document(page_content=content,
                                metadata=metadata,
                                id=str(i)
                                )
            documents.append(document)
            ids.append(str(i))
        self.docs['docs'] = documents
        self.docs['ids'] = ids

class AICodeVector:
    def __init__(self, codeDir, dbLocation, collectionName="codebase", ragType="code", model="llama3"):
        self.codeDir = codeDir
        self.dbLocation = dbLocation
        self.collectionName = collectionName
        self.ragType = ragType
        self.docs = None
        self.vectorStore = None
        self.ragChain = None
        self.model = model
        self.prompt = None
        self.setupPrompt()

    def setupPrompt(self, system_prompt=None):
        """Sets up the prompt for the LLM. Use a default if none provided."""
        if system_prompt is None:
            if self.ragType == "code":
                system_prompt = (
                    "You are an expert Python software engineer. Use only the provided python codebase "
                    "and your Python pep-8 language knowledge to answer the questions asked."
                    "The context provided is included within the Python codebase."
                    "Use any and available, except AI* references, in the Python codebase to answer the question"
                    "The use of data references in the codebase are also allowed."
                    "Do not make up information. If the answer is not in the above context."
                    "\n\n"
                    "{context}"
                )
            else:
                system_prompt = (
                    "You are an expert in answering questions about processors and Python Dash apps. Answer as concisely as possible."
                    "Models: {models}"
                    "Question: {question}"
                )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )


    def loadCodebase(self):
        print("Loading codebase...")
        self.docs = self.codebaseLoader()
        print(f"Loaded {len(self.docs)} documents.")
        print("Splitting documents...")
        self.chunks = self.splitDocs()
        print(f"Split into {len(self.chunks)} chunks.")
        print("Setting up vector store...")
        self.vectorStore = self.setupVectorStore()
        print("Vector store ready.")
        print("Setting up RAG chain...")
        self.ragChain = self.setupRagChain()
        print("RAG chain ready. You can now ask questions about your codebase.")
        return self.ragChain

    def codebaseLoader(self):
        """Loads all text files (including .py) from a directory."""
        # Use TextLoader with a file extension filter for code files
        loader = DirectoryLoader(
            self.codeDir,
            glob="**/*.py", # Targets only .py files
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
        )
        docs = loader.load()
        self.docs = docs
        return docs

    def splitDocs(self):
        """Splits documents into smaller, manageable chunks."""
        # RecursiveCharacterTextSplitter is good for maintaining context in code
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language="python",
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(self.docs)
        self.chunks = chunks
        return chunks

    def setupVectorStore(self):
        """Creates embeddings and stores them in ChromaDB."""
        # The model name should match the one pulled with Ollama
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # Use Chroma as the vector store
        self.vectorstore = Chroma.from_documents(
            collection_name=self.collectionName,
            documents=self.chunks,
            embedding=embeddings,
            persist_directory=dbLocation
        )
        # vectorStore = Chroma(collection_name="processors", embedding_function=embeddings, persist_directory=dbLocation)

    def setupRagChain(self):
        """Sets up the RAG chain using Ollama and LangChain."""
        # Initialize the local LLM
        llm = Ollama(model=self.model)
        retriever = self.vectorstore.as_retriever()
        # Create the chains
        question_answer_chain = create_stuff_documents_chain(llm, self.prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

if __name__ == "__main__":
    vector = AICodeVector(codeDir, dbLocation, "llama3.2")
    ragChain = vector.loadCodebase()

