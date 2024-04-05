import wolframalpha

#question=input(">>aknkankxna")
client=wolframalpha.Client("GWRXQ7-9XPH9YKY9Q")

def  wolframalpha_result(question):
    res=client.query(question)
    answer=next(res.results).text
    print(answer)

    