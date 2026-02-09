""" Working with AI Coding agents """

from datetime import datetime

from AI.vectorCoder import AICodeVector



codeDir = '/home/wrichter/Documents/Code/Projects/Python/processors'
dbLocation = "./AI/chromeLangChainCodeDb"
startTime = datetime.now()
vector = AICodeVector(codeDir, dbLocation, "llama3.2")
ragChain = vector.loadCodebase()
endTime = datetime.now()
print(f"Load Time: {endTime - startTime}")

while True:
    print('\n\n----------------------------------------------------------------------------"')
    query = input('Ask a Code Question (q to quit):\nq')
    print('\n\n----------------------------------------------------------------------------"')
    if query.lower() == 'q':
        break
    startTime = datetime.now()
    response = ragChain.invoke({"input": query})
    endTime = datetime.now()
    print("\n--- Answer ---")
    print(response["answer"])
    print("--------------\n")
    print(f"Lookup Time: {endTime - startTime}")
