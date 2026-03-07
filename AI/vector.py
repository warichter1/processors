""" AI vector search"""

from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

from nvidia import nvidiaLoader, nvidiaHeader, nvidiaSelect
from arm import armLoader, armSelect # ArmDataProcessor
from dashConf import dashConfig as conf, dashStyles as style
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

embeddings = OllamaEmbeddings(model='mxbai-embed-large')
dbLocation = "./AI/chromeLangChainDb"
addDocuments = not os.path.exists(dbLocation)
vectorStore = Chroma(collection_name="processors", embedding_function=embeddings, persist_directory=dbLocation)
if addDocuments:
    df = importProcessors(debug=False)
    documents = []
    ids = []
    for i, row in df.iterrows():
        print(i)
        document = Document(page_content=row['manufacturer'] + " " + row['processor_family'] + " " + row['microarchitecture'] + " " + str(row['code_name']) + " " + str(row['model']),
                            metadata={'cores': str(row['hw_ncores']), 'threadspercore': str(row['hw_nthreadspercore']), 'date': str(row['created_at'])},
                            id=str(i)
                            )
        documents.append(document)
        ids.append(str(i))
    vectorStore.add_documents(documents=documents, ids=ids)
retriever = vectorStore.as_retriever(search__kwargs={'k': 5})
