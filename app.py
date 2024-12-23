import pandas as pd
import streamlit as st
#from review_summary import get_review_summary
from review_summary_gemini import get_review_summary

# displaying page title and header
st.title("E-Commerce Customer Review Summarization Engine")
st.header("Below is the Data we have from Kaggle")

data = pd.read_csv("data/Musical_instruments_reviews.csv", usecols = ['reviewerID', 'asin', 'reviewText', 'overall', 'summary', 'reviewTime'])

st.dataframe(data)

unique_product_list = data["asin"].unique()

# develop form
with st.form(key="user_interaction"):

    #select your product input option
    input_options = st.empty()

    submit_button = st.form_submit_button("Submit")

# setting up product query fields based on the input selection option
with input_options: 

    input_opt = st.selectbox(
        label = "Please select your product selection option.",
        options = unique_product_list
    )


data.columns = [column.replace(" ", "_") for column in data.columns]

data.rename(columns={"overall":"rating"}, inplace=True)

products = data["asin"].unique().tolist()

product_asin = input_opt

if submit_button:
    
    with st.spinner("Processing your data now...."):
        # getting outputs now 
        tokens, df_reviews, summary_small, summary_map, summary_refine = get_review_summary(data, str(input_opt))
        
        # displaying customer reviews in dataframe format 
        df = df_reviews.head(10)
        st.markdown(" ### Top 10 Customer Reviews: \n")
        st.dataframe(df, hide_index=True)

    st.success("Data Processing Complete!")

    if tokens <= 3500:
        
        # displying small customer reviews summary
        st.markdown(" ### The customer reviews can be summarized as: \n")
        st.write(summary_small.content)

    else:
        # for larger review content
        st.markdown(" ##### The Customer Reviews content is large. It warrants the use of 'Map Reduce' and 'Refine Method' for summary generation. \n")
        
        # displaying Map Reduce customer review summary
        st.markdown(" ### The customer reviews summary using Map Reduce Method: \n")
        st.write(summary_map["output_text"])

        # displaying Refine Method customer review summary
        st.markdown(" ### The customer reviews summary using Refine Method: \n")
        st.write(summary_refine["output_text"])

