""" AI vector search"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from nvidia import nvidiaLoader, nvidiaHeader, nvidiaSelect
from arm import armLoader, armSelect # ArmDataProcessor
from cpu import loadProcessors

def importProcessors(debug=False):
    processors = loadProcessors(debug=False)
    amdDf = processors.selectManufacturer('AMD')
    amdDf[' index'] = range(1, len(amdDf) + 1)
    intelDf = processors.selectManufacturer('Intel')
    intelDf[' index'] = range(1, len(intelDf) + 1)
    otherDf = processors.selectManufacturer('notintelamd')
    otherDf[' index'] = range(1, len(otherDf) + 1)
    nvidiaDf = nvidiaLoader('nvidia', columns=nvidiaSelect)
    nvidiaDf[' index'] = range(1, len(nvidiaDf) + 1)
    armDf = armLoader('ARM', columns=armSelect)
    armDf[' index'] = range(1, len(armDf) + 1)
    df = processors.fullDf
    return df


class AICodeLoader:
    def __init__(self, codeDir, ragType="code", model="qwen3-coder", keys=None, metadata=None, collectionName="codebase",
                 dbLocation=None, prompt=None):
        print('Initialize Code Loader')
        self.codeDir = codeDir
        self.dbExists = False
        self.ragType = ragType
        self.model = model
        self.dbLocation = dbLocation
        self.collectionName = collectionName
        self.prompt = prompt
        self.keys = keys
        self.metadata = metadata
        self.docs = dict(docs=None, ids=None)
        self.loadCodebase()

    def loadCodebase(self):
        print('Load code base')
        loader = DirectoryLoader(
            self.codeDir,
            glob="**/*.py", # Targets only .py files
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
        )
        self.splitDocs(loader.load())

    def splitDocs(self, docs):
        """Splits documents into smaller, manageable chunks."""
        print('Split documents')
        # RecursiveCharacterTextSplitter is good for maintaining context in code
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language="python",
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        self.docs['docs'] = chunks


class AIDfLoader:
    """
        Loads a DataFrame as documents for the vector storage.
        Used as a Data import for the RAG vector.
    """
    def __init__(self, df, ragType="dataframe", model="llama3", keys=None, metadata=None, prompt=None,
                 collectionName="processors", dbLocation = "./AI/chromeLangChainDb2"):
        print('Intialize Dataframe Loader')
        self.docs = dict(docs=None, ids=None)
        self.ragType = ragType
        self.model = model
        self.prompt = prompt
        self.codeDir = None
        self.dbLocation = dbLocation
        self.collectionName = collectionName
        self.dbExists = not os.path.exists(dbLocation)
        self.keys = keys if keys else ['manufacturer', 'processor_family', 'microarchitecture', 'code_name', 'model']
        self.metadata = metadata if metadata else {'cores': 'hw_ncores', 'threadspercore': 'hw_nthreadspercore',
                                                    'date': 'created_at'}
        if self.dbExists:
            self.loadDocuments(df)

    def loadDocuments(self, df):
        """Loads a DataFrame as  if the vector store does not exist."""
        print('Load Docs')
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


class AiCodeVector:
    def __init__(self, inputRag):
        print('Initialize AI Code Vector')
        self.codeDir = inputRag.codeDir
        self.dbLocation = inputRag.dbLocation
        self.dbExists = inputRag.dbExists
        self.collectionName = inputRag.collectionName
        self.metadata = inputRag.metadata
        self.ragType = inputRag.ragType
        self.docs = inputRag.docs
        self.vectorStore = None
        self.retrievalChain = None
        self.model = inputRag.model
        self.prompt = inputRag.prompt
        self.setupVector = dict(code=self.setupVectorStoreCode, dataframe=self.setupVectorStoreData)
        self.setupPrompt()
        self.setupVector[self.ragType]()
        self.setupRagChain()

    def setupPrompt(self):
        """Sets up the prompt for the LLM. Use a default if none provided."""
        print('Setup Prompt')
        if self.prompt is None:
            if self.ragType == "code":
                print('Code Prompt')
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
            elif self.ragType == "dataframe":
                print('Dataframe Prompt')
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

    def setupVectorStoreCode(self):
        """Creates embeddings and stores them in ChromaDB."""
        print('Setup Vector Store Code')
        # The model name should match the one pulled with Ollama
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # Use Chroma as the vector store
        self.vectorStore = Chroma.from_documents(collection_name=self.collectionName,
                                                documents=self.docs['docs'],
                                                embedding=embeddings,
                                                persist_directory=self.dbLocation
                                                )

    def setupVectorStoreData(self):
        """Creates embeddings and stores them in ChromaDB."""
        print('Setup Vector Store Data')
        embeddings = OllamaEmbeddings(model='mxbai-embed-large')
        self.vectorStore = Chroma(collection_name=self.collectionName, embedding_function=embeddings,
                             persist_directory=self.dbLocation)
        if self.dbExists:
            # self.docs = dict(docs=None, ids=None)
            self.vectorStore.add_documents(documents=self.docs['docs'], ids=self.docs['ids'])
        # retriever = vectorStore.as_retriever(search__kwargs={'k': 5})
        # self.vectorStore = vectorStore

    def setupRagChain(self):
        """Sets up the RAG chain using Ollama and LangChain."""
        print('Setup RAG Chain')
        # Initialize the local LLM
        llm = Ollama(model=self.model)
        retriever = self.vectorStore.as_retriever(search__kwargs={'k': 5})
        # Create the chains
        question_answer_chain = create_stuff_documents_chain(llm, self.prompt)
        self.retrievalChain = create_retrieval_chain(retriever, question_answer_chain)
        print("Setup Complete")

    def getRagChain(self, question):
        """Ask a Question of the rag chain."""
        print(f'Get RAG Chain: {question}')
        return self.retrievalChain.invoke({"input": question})

