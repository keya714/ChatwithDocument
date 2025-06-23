from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
import yaml
import box
import timeit



with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

if __name__ == "__main__":

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2', model_kwargs={'device': 'cpu'})
    connection_string = "postgresql://postgres:keya123@172.31.245.200:5433/postgres"

    # Define the prompt template for the LLM
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
        Use the following documents to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {context}
        Answer:
        """,
        input_variables=["question", "context"],
    )

    # Initialize the LLM with Llama 3.1 model
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )

    vector_store = PGVector(
    embeddings=embeddings,
    collection_name='csvfiles',
    connection=connection_string,
    use_jsonb=True,
    )

    query=input("Enter your query:\n")
    while query != "end":
        start = timeit.default_timer()
        retriever = vector_store.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT})
        
        # Create a new retrieval chain without the filter
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
            chain_type_kwargs={'prompt': prompt}
        )
        
        # Perform the search again without the metadata filter
        response = qa_chain.invoke({'query': query})

        end = timeit.default_timer()
        print(response)
        print("#"*30)
        print(response['result'])
        print(f"Time to retrieve answer: {end - start}")
        print("#"*30)
        query=input("Enter your query:\n")