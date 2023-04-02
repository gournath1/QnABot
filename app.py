import os
os.environ["OPENAI_API_KEY"] =  ${{ secrets.OAPI_KEY }}

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("Gita.pdf")

from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

embeddings = OpenAIEmbeddings()
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])
llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

def chatbot_response(query):
    docs = docsearch.vectorstore.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__ == '__main__':
    app.run()
