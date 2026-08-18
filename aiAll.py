"""
    Working with AI agents
"""
from datetime import datetime

from AI.vectorCoder import AIDfLoader, AICodeLoader, importProcessors, AiCodeVector

codeDir = '/home/wrichter/Documents/Code/Projects/Python/processors'

if __name__ == '__main__':
    # codeLoader = AICodeLoader(codeDir, dbLocation="./AI/chromeLangChainCodeDb")
    # dfLoader = AIDfLoader(df, dbLocation="./AI/chromeLangChainDb2")
    # aiCoder = AiCodeVector(codeLoader)
    # aiData = AiCodeVector(dfLoader)
    aiCoder = None
    aiData = None
    while True:
        print('\n\n----------------------------------------------------------------------------"')
        print('1: AI Coder')
        print('2: AI Data')
        question = input('Select 1 or 2 (q to quit): ')
        print('\n\n----------------------------------------------------------------------------"')
        match question.lower():
            case 'q':
                exit
            case "1":
                codeLoader = AICodeLoader(codeDir, dbLocation="./AI/chromeLangChainCodeDb")
                aiCoder = AiCodeVector(codeLoader)
                print("Begin Coder Prompt.")
                promptType = "Code"
                break
            case "2":
                df = importProcessors(debug=False)
                metadata = {'models': ['Nvidia', 'AMD', 'Intel', 'ARM','Zilog', 'Motorola']}
                dfLoader = AIDfLoader(df, dbLocation="./AI/chromeLangChainDb2", metadata=metadata)
                aiData = AiCodeVector(dfLoader)
                print("Begin Processor Data Prompt.")
                promptType = "Dataframe"
                break
            case _:
                print("Please Select a valid option.")
    while True:
        print('\n\n----------------------------------------------------------------------------"')
        query = input(f'Ask a {promptType} Question (q to quit):\n')
        print('\n\n----------------------------------------------------------------------------"')
        if query.lower() == 'q':
            break
        startTime = datetime.now()
        if question == '1':
            result = aiCoder.getRagChain(query)  # .invoke({"input": query})
            print(result)
        elif question == '2':
            # result = aiData.getRagChain().invoke({"input": query, 'models': ['Nvidia', 'AMD', 'Intel', 'ARM',
            #                                                                    'Zilog', 'Motorola']})
            result = aiData.getRagChain(query)  #.invoke({"input": query, 'models': ['Nvidia', 'AMD', 'Intel', 'ARM','Zilog', 'Motorola']})
            print(result)

