import streamlit as st
from langchain.prompts import PromptTemplate #prompt engineering
from langchain_community.llms import CTransformers    #used to call ggml models 

##Function to get response from LLama-2 model

def get_response(input_text,word_limit):
    # Call Llama2 model
    llm = CTransformers(model='models\llama-2-7b-chat.ggmlv3.q6_K.bin', 
                        model_type = 'llama',
                        config = {'max_new_tokens':256,
                                  'temperature':0.001})
                        #tokens can be tweaked according to the required blog length 
    #Prompt Engineering

    template = """
    You are a professional blogger and your task is to write a blog post on the given topic: {input_text}. The article must be close to {word_limit} words. The article should be be lengthy. 
    Give it as plain text.
    """

    prompt = PromptTemplate(input_variables=["input_text","word_limit"],
                            template=template)
    
    # Generate response from LLama2 Model
    response = llm.invoke(prompt.format(input_text = input_text,word_limit = word_limit))
    
    print(response)
    return response

    # return response



st.set_page_config(page_title="Generate Blogs",
                   page_icon="🤖🖊️🖊️📝",
                   layout="centered",
                   initial_sidebar_state="collapsed")
st.header('Generate your blog 🤖🖊️🖊️📝')

#Creating columns for input, word_limit and 
input_text = st.text_input('Enter your topic')

col1,col2 = st.columns([5,5])

with col1:
    word_limit = st.text_input('Number of Words')

submit = st.button("Generate")

if submit:
    st.write(get_response(input_text,word_limit))
    