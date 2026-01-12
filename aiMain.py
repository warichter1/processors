""" Working with AI agents """

import pandas as pd
from datetime import datetime
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from AI.vector import retriever


model = OllamaLLM(model='llama3.2')
template = """You are an expert in answering questions about processors and Python Dash apps. Answer as concisely as possible.
               Models: {models}
               Question: {question}"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print('\n\n----------------------------------------------------------------------------"')
    question = input('Ask a CPU Question (q to quit): ')
    print('\n\n----------------------------------------------------------------------------"')
    if question.lower() == 'q':
        break
    startTime = datetime.now()
    response = retriever.invoke(question)
    result = chain.invoke({'question': {'question': question}, 'models': ['Nvidia', 'AMD', 'Intel', 'ARM', 'Zilog', 'Motorola']})
    print(result)
    endTime = datetime.now()
    print(f"Lookup Time: {endTime - startTime}")
