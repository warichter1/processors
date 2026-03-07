"""
    Working with AI agents
"""
from datetime import datetime

from AI.vectorCoder import AIDfLoader, AICodeLoader, importProcessors, AiCodeVector

df = importProcessors(debug=False)
codeDir = '/home/wrichter/Documents/Code/Projects/Python/processors'

if __name__ == '__main__':
    codeLoader = AICodeLoader(codeDir, dbLocation="./AI/chromeLangChainCodeDb")
    dfLoader = AIDfLoader(df, dbLocation="./AI/chromeLangChainDb2")

    aiCoder = AiCodeVector(codeLoader)
    aiData = AiCodeVector(dfLoader)

    while True:
        print('\n\n----------------------------------------------------------------------------"')
        question = input('Select 1 or 2 (q to quit): ')
        print('1: AI Coder')
        print('2: AI Data')
        print('\n\n----------------------------------------------------------------------------"')
        if question.lower() == 'q':
            exit()
        elif question in ['1', '2']:
            continue
    while True:
        print('\n\n----------------------------------------------------------------------------"')
        query = input('Ask a Code Question (q to quit):\nq')
        print('\n\n----------------------------------------------------------------------------"')
        if query.lower() == 'q':
            break
        startTime = datetime.now()
        if question == '1':
            result = aiCoder.setupRagChain().invoke({'question': query, 'models': ['Nvidia', 'AMD', 'Intel', 'ARM', 'Zilog', 'Motorola']})
            print(result)
        elif question == '2':
            result = aiData.setupRagChain().invoke({"input": query})
            print(result)

