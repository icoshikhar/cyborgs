from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from typing import Any
import pandas as pd
import tiktoken

from dotenv import load_dotenv 
import os 

load_dotenv()

# importing api keys and initiate llm
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

'''

#Define the Summarize Chain
summary_statement = """You are an expeienced copy writer providing a world-class summary of product reviews {cust_reviews} from numerous customers \
                        on a given product from different leading e-commerce platforms. You write summary in 80 words of all reviews for a target audience \
                        of wide array of product reviewers ranging from a common man to an experienced product review professional."""

summary_prompt = ChatPromptTemplate.from_template(template=summary_statement)

summary_chain = summary_prompt | llm

#Load the comments


docs = "What are products of Flipkart"

result = summary_chain.invoke(docs)
print(result.content)

'''

# Counting Review output tokens
def count_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# split large reviews
def document_split(cust_reviews: str, chunk_size: int, chunk_overlap: int) -> Any:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""])
    
    # converting string into a document object
    docs = [Document(page_content = t) for t in cust_reviews.split('\n')]
    split_docs = text_splitter.split_documents(docs)
    return split_docs


# Applying map reduce to summarize large document
def map_reduce_summary(split_docs: Any) -> str: 
    map_template = """Based on the following docs {docs}, please provide summary of reviews presented in these documents. 
    Review Summary is:"""

    map_prompt = ChatPromptTemplate.from_template(map_template)
    map_chain = map_prompt | llm

    # Reduce
    reduce_template = """The following is set of summaries: 
    {doc_summaries}
    Take these document and return your consolidated summary in a professional manner addressing the key points of the customer reviews. 
    Review Summary is:"""
    reduce_prompt = ChatPromptTemplate.from_template(reduce_template)

    # Run chain
    reduce_chain = reduce_prompt | llm

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    #ccombine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="doc_summaries")
    combine_documents_chain = create_stuff_documents_chain(llm, reduce_prompt, document_variable_name="doc_summaries")

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=3500,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )
    
    # generating review summary for map reduce method
    cust_review_summary_mr = map_reduce_chain.invoke(split_docs)

    return cust_review_summary_mr


def refine_method_summary(split_docs) -> str:
    prompt_template = """
                  Please provide a summary of the following text.
                  TEXT: {text}
                  SUMMARY:
                  """

    question_prompt = ChatPromptTemplate(
        template=prompt_template
    )

    refine_prompt_template = """
                Write a concise summary of the following text delimited by triple backquotes.
                Return your response in that covers the key points of the text.
                ```{text}```
                BULLET POINT SUMMARY:
                """

    refine_prompt = ChatPromptTemplate(
        template=refine_prompt_template)

    # Load refine chain
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=False,
        input_key="input_text",
        output_key="output_text",
    )
    
    # generating review summary using refine method
    cust_review_summary_refine = chain.invoke({"input_text": split_docs}, return_only_outputs=True)
    return cust_review_summary_refine


# generate review summary for smaller revieww
def small_reviews_summary(cust_reviews: str) -> str:
    summary_statement = """You are an expeienced copy writer providing a world-class summary of product reviews {cust_reviews} from numerous customers \
                        on a given product from different leading e-commerce platforms. You write summary of all reviews for a target audience \
                        of wide array of product reviewers ranging from a common man to an expeirenced product review professional."""
    summary_prompt = ChatPromptTemplate.from_template(template=summary_statement)
    summary_chain = summary_prompt | llm
    review_summary = summary_chain.invoke(cust_reviews)
    return review_summary


def get_review_summary(reviews_data, productId: str) -> tuple[int, Any, str, str, str]:

    data = pd.DataFrame(reviews_data)
    data.columns = [column.replace(" ", "_") for column in data.columns]
    data.rename(columns={"overall":"rating"}, inplace=True)
    data.sort_values(by=["rating"], ascending=False, inplace=True)
    data.query(f'asin == "{productId}"', inplace=True)
    reviews = " ".join(each for each in data.reviewText)

    # Checking review length
    total_tokens = count_tokens(str(reviews), "cl100k_base")

    if total_tokens <= 3500:
        cust_review_summary = small_reviews_summary(reviews)
        cust_review_summary_map = "N.A."
        cust_review_summary_refine = "N.A."
    else:
        split_docs = document_split(reviews, 1000, 50)
        cust_review_summary_map = map_reduce_summary(split_docs)
        cust_review_summary_refine = refine_method_summary(split_docs)
        cust_review_summary = "N.A."

    return total_tokens, data, cust_review_summary, cust_review_summary_map, cust_review_summary_refine