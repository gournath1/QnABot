
import os
os.environ["OPENAI_API_KEY"] = "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"

from google.colab import files
uploaded = files.upload()

from langchain.document_loaders import TextLoader
loader = TextLoader('Gita.txt')

from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

#query = "Who is Gour ?"
#index.query(query)

from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

embeddings = OpenAIEmbeddings()
#docsearch = Chroma.from_documents(texts, embeddings)
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])
llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

import textwrap
import colorama
from colorama import Fore
wrapper = textwrap.TextWrapper(width=100)
while True:
  query = input(Fore.RED + "Ask me ...")
  if query=="stop":
     print("Good Bye")
     break
  docs = docsearch.vectorstore.similarity_search(query)
  response = chain.run(input_documents=docs, question=query)
  word_list = wrapper.wrap(text=response)
  for element in word_list:
    print(Fore.GREEN + element)
