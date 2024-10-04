#Importing neccesary libraries
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import StuffDocumentsChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tabula
import numpy as np
import os

# Setting up our env
from dotenv import load_dotenv
load_dotenv()
output_summary=""
#Getting the Hugging Face Token and GroqApiKey
# os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
groq_api_key=st.secrets['GROQ_API_KEY']
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#Setting up streamlit
st.title("Identifying Red flags in a SAAS document")
st.write("Upload pdf to see the red flags")
# temperature=st.slider("Set your temperature as you require",0.0,1.0,0.7)

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-70b-versatile",temperature=0.4)

#Now lets start with getting the session id 
session_id=st.text_input("Enter your session id",value='default_session')

#Stateful management of Chat History
if 'store' not in st.session_state:
    st.session_state.store={}


#Uploading the pdf
uploaded_documents=st.file_uploader("Choose A PDf file to upload",type="pdf",accept_multiple_files=True)
if uploaded_documents:
    documents=[]
    red_flag_task=0
    for uploaded_document in uploaded_documents:
        temp_pdf=f"./temp.pdf" 
        with open(temp_pdf,'wb') as file:
            file.write(uploaded_document.getvalue())
            file_name=uploaded_document.name
        loader=PyPDFLoader(temp_pdf)
        docs=loader.load()
        documents.extend(docs)
    user_input=st.text_input("Enter your question that you want to ask from Janie")
    button_input=st.button("Tell me the red flags")

    if button_input:
        red_flag_task=1
        if output_summary!="":
            st.write(output_summary)
            red_flag_task=2
        
    elif user_input:
        red_flag_task=0
    


    
    #Taking the input from the user 
    #This is the whole point where the bifurcation between a rag application and a red flag detector starts
    
    #Similarity function 
    # cosine_arr=[]
    # def get_similarity_score(user_input, valid_prompts):
    #     for prompt in valid_prompts:
    #         embedding1=embeddings.embed_query(user_input)
    #         embedding2=embeddings.embed_query(prompt)
    #         cosine_similarity = np.dot(embedding1,embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    #         cosine_arr.append(cosine_similarity)
    #     best_score=max(cosine_arr)
    #     return best_score
    
    # similarity_threshold = 0.5
    # valid_prompts = [
    # "Identify the red flags in this document",
    # "Detect red flags",
    # "Analyze document for red flags",
    # "Find the red flags",
    # "Tell me things that I should be aware of before buying from this vendor",
    # "Flag any potential issues in this document",
    # "What warnings or red flags do you see in this document?",
    # "Tell me some clauses of these documents that are not in favour of me"
    # "Tell me important things from the document that can help me make a good and an informed decision"
    # ]
    
    # red_flag_task=1
    # if user_input:
    #     similarity_score = get_similarity_score(user_input, valid_prompts)
    #     st.write(similarity_score)
    #     if similarity_score >= similarity_threshold:
    #         red_flag_task=1
    #         #Valid prompts to trigger the red flag generation
    #     else:
    #         red_flag_task=1

    #General task
    # Improve the chunking technique first to making chunks para wise 
    text_splitter3=RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=200)
    chunks3=text_splitter3.split_documents(documents) 
    
    if red_flag_task==1:
        st.write("I am working ")
        system_prompt = (
                "You are an assistant specialized in flaging the redline for the customer"
                "Your task is to help people make an informed decision"
                "You will be given a piece of information and you will be required to tell what are the things that a customer should be aware of before signing the deal"
                "If you find thing that should be known to the customer for making an informed decision just output that"
                "Output only those things from the terms and condition document that you think must be known to the customer for making an informed decision or things that are not in favour of the customer"
                )
        qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{context}"),
                    ]
                )
        # qa_chain = load_qa_chain(llm, chain_type="stuff")
        question_answer_chain=create_stuff_documents_chain(llm=llm,prompt=qa_prompt,document_variable_name="context")
        Whole_document=[]
        for i in range(len(chunks3)):
            doc=[Document(page_content=str(chunks3[i].page_content))]
            response = question_answer_chain.invoke({"context": doc})
            Whole_document.append(response)
        cleandocs=[]
        for i in range(len(Whole_document)):
            cleandocs.append(Document(page_content=str(Whole_document[i])))
        val=3000//len(Whole_document)

        chunks_prompt="""
        Please summarize the below Terms and Condition Document in less than X words:
        Document:'{text}'
        Summary:

        """

        list1=[elm for elm in chunks_prompt]
        for i in range(len(list1)):
            if list1[i]=='X':
                list1[i]=val
        chunks_prompt1="".join([str(elm) for elm in list1])



        map_prompt_template=PromptTemplate(input_variables=['text'],template=chunks_prompt1)
        final_prompt='''
        Provide the final summary of the entire Terms and conditions that should be highlighted to the cutomer for taking an informed decision.
        Output only those things from the terms and condition document that you think must be known to the customer for making an informed decision or things that are not in favour of the customer
        Summmary should only contain 2 components that includes points to keep in mind for an informed decision and clauses that are not mentioned in the document
        Summary should consist of 20 most important points to keep in mind for an informed decision.
        Apart from that summary should also include atleast 15 clauses that are not mentioned in the document.
        summary must not contain the heading like Clauses That Are Not in Favor of the Customer
        Also if the things that are mentioned below are not mentioned in the document then add that in the summary that there is no mention of this clause in the document and please review it.
        these are in the format like clause description - clause that customer should aim for.
        Below are some things which are in favour of the customer and should be aimed for.
        \n
        License Scope-A broader scope, to include possible use by subsidiaries, affiliates and contractors. Fewer license restrictions, that are fair and reasonable. "
        Payment Term-Payment in arrears. Longer payment terms with right to dispute payments in good faith (e.g., net 60 after receipt of undisputed invoice). Avoid interest and penalties; or minimize their impact via a written notice requirement and cure periods before any interest or penalties can begin."
        Service level agreement (SLA)- Robust SLAs, including a right to service credits or refunds for excessive downtime, accessible support channels as well as a right to terminate after a certain number (or length) of incidents."
        Use of Data/Data Rights-Retain all rights to its data; or grant limited rights to vendor for the use aggregated and anonymized data only."
        DPA (data privacy addendum)-DPA that requires prompt vendor notice (e.g., 48 hours) in the event of not only an actual security breach, but also any suspected or alleged security breaches; quick remediation (at vendor expense); termination rights for customer; and indemnity for security breach with either unlimited liability or a higher super-cap."
        Reps and Warranties-Standard, but broader, vendor reps and warranties, e.g., that vendor will comply with applicable laws and industry standards, confidentiality and privacy protections, IP rights (non-infringement), etc."
        Indemnities-No indemnities given to vendor; or give indemnities with a narrow scope (and include exceptions for modification or misuse of your content or data). Robust indemnities from vendor (e.g., non-infringement, confidentiality & privacy, injury to persons or property, arising from any material breach, etc.)."
        Limitation on Liability-Uncapped vendor liability, if possible, especially for issues such as indemnities, IP violations, and confidentiality or privacy breaches. May accept super caps if they are reasonable, based on the scope of possible harm, not necessarily proportional to the size of the deal"
        Termination Rights-Broad termination rights (e.g., due to vendor breach, SLA failures, privacy issues, decrease in service features or functionality, chronic issues, and, if possible, for convenience); with rights to pro-rata refund, if possible."
        Renewal-Auto renewal may be acceptable, but only with reasonable opt out dates for customer to avoid paying for an unwanted renewal term."
        Notice Period-Preferred length of notice periods and timelines also varies. Shorter notice requirements for things relating to customer rights. Longer notice periods for any provisions giving the vendor a right to pursue remedies against customer."
        Insurance-Vendor insured for general liability, errors & omissions/professional liability, cyber liability, and workmen's comp. Plus, an umbrella policy and other applicable coverage based on circumstances (car, shipping, air, etc.)."
        Pubilicity-Right to approve any use of customer name or logos, including prior approval of use in lists of clients."
        Assignment-Mutual restriction of assignment, with a mutual exception for Mergers and Acquisitions activity or reorganizations."
        Pricing Plan-A nominal or economical pricing plan"
        Others-If asked to sign vendor template contract, review for non-standard terms to avoid, such as Exclusivity ,non-solicitation clauses,Liens and security interests or anything else unusual or non-standard."
        \n

        Document:{text}
        '''
        final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)
        summary_chain=load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=final_prompt_template,
        verbose=True
        )
        output_summary=summary_chain.run(cleandocs)

        st.write(output_summary)
        # st.write(chunks_prompt)
    elif red_flag_task==0:
        vector_store=FAISS.from_documents(documents=chunks3,embedding=embeddings)
        faiss_semantic_retriever=vector_store.as_retriever()
        bm25_retriever = BM25Retriever.from_documents(documents=chunks3)
        retriever = EnsembleRetriever(retrievers=[bm25_retriever,faiss_semantic_retriever],weights=[0.5,0.5])
        context_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    
        context_prompt=ChatPromptTemplate.from_messages(
                    [
                        ("system", context_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )

        #Creating a history aware retreiver
        #Combines both past information and context retreived information and fomulates it in one question
        #Output is a set of documents
        history_aware_retriever=create_history_aware_retriever(llm,retriever,context_prompt)

        #Question answering prompt creating task starts here
        system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use four sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"
                )
        qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
        
        #Create a chain for passing a list of Documents to a model.
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        #Binding 2 chains together
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        #Creating storage functionality
        def get_session_history(session:str)->BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id]=ChatMessageHistory()
                return st.session_state.store[session_id]

        conversational_rag_chain=RunnableWithMessageHistory(
                rag_chain,get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
        if user_input:
                session_history=get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id":session_id}
                    },  # constructs a key "abc123" in `store`.
                )
                # st.write(st.session_state.store)
                st.write("Assistant:", response['answer'])
                # st.write("Chat History:", session_history.messages)
