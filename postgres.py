from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain.schema import Document
import box
import yaml
import re
import spacy

with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def load_pdfs_from_folder(folder_path):
    pdf_loader = DirectoryLoader(folder_path, glob='*.pdf', loader_cls=PyPDFLoader)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP, length_function=len, add_start_index=True)
    documents = pdf_loader.load()
    nlp = spacy.load("en_core_web_sm")

    doc = ""
    for i in documents:
        # print(i, type(i))
        doc += i.page_content
        # print(doc)
        # break
        # doc.metadata["id"] = doc.metadata.get("id", None) 
    doc = re.sub(r'\s+', ' ', doc)  # Replace multiple spaces with one
    doc = doc.replace("\n", " ")  # Replace newlines with a single space
    print(doc)
    docs = text_splitter.create_documents([doc])
    print(docs[-1])
    for i, chunk in enumerate(docs):
        chunk.metadata['keywords']=extract_keywords_pos(chunk.page_content, nlp, 10)
        # print(f"Chunk {i}, Start Index: {chunk.metadata['start_index']}, Content Preview: {chunk.page_content[:500]}")
    return docs



def load_csv(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    documents = []
    for index, row in df.iterrows():
        # Combine all columns into a single string or pick specific columns
        content = ' '.join(str(value) for value in row.values)
        # Create a Document object
        documents.append(Document(page_content=content, metadata={"row_index": index}))
    return documents




def extract_keywords_pos(string, nlp, n_keywords=5):
    
    doc = nlp(string)
    
    # Extract nouns and proper nouns as keywords
    keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
    
    # Return the most frequent nouns
    keyword_freq = {word: keywords.count(word) for word in set(keywords)}
    sorted_keywords = sorted(keyword_freq, key=keyword_freq.get, reverse=True)
    return sorted_keywords[:n_keywords]




if __name__ == "__main__":

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2', model_kwargs={'device': 'cpu'})
    connection_string = "postgresql://postgres:keya123@172.31.245.200:5433/postgres"



    vector_store = PGVector(
    embeddings=embeddings,
    collection_name='csvfiles',
    connection=connection_string,
    use_jsonb=True,
    )

    # docs=load_pdfs_from_folder('data/')
    docs=load_csv(r'C:\Users\keyar\Documents\Internships\Oneclick\Tasks\LangChain\data\flights.csv')
    print(f"Number of documents to be added to PostgreSQL: {len(docs)}")
    vector_store.add_documents(docs, ids=[id for id in range(len(docs))])

# path=r"C:\Users\keyar\Documents\Internships\Oneclick\Tasks\LangChain\data"
# load_pdfs_from_folder(path)

# #pdf/doc
# #name
# #mobile number
# #email
# #skills(programming language)