from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq 
from langserve import add_routes
import os 
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Define the prompt template
generic_template = "Transalte the following question to {language} language"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("human", "{question}"),
    ]
)


#create the output parser
parser = StrOutputParser()

#create the chain
chain = prompt | model | parser

#App defination
app = FastAPI(title="LangchainServer",
              description="This is a demo server for langchain with groq model",
              version="1.0.0")

#add chain route to the app
add_routes(app,chain,path="/chain")

#main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

