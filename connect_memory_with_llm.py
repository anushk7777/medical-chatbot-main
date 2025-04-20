import os
import subprocess
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Install TTS library (SpeechT5) and Translation libs
try:
    subprocess.run(
        ["pip", "install", "--timeout", "60", "--retries", "3", "-q", "transformers", "torch", "scipy"],
        check=True,
        capture_output=True,
    )
    print("SpeechT5, Translation libraries, and scipy installed successfully!")
    tts_installed = True  # Flag to indicate successful installation
    translation_installed = True
except subprocess.CalledProcessError as e:
    print(f"Error installing SpeechT5 and Translation libraries: {e.stderr.decode()}")
    tts_installed = False  # Set the flag to False in case of installation failure
    translation_installed = False

# Attempt to import the TTS and Translation libraries after installation
if tts_installed and translation_installed:
    try:
        from transformers import pipeline
        import scipy.io.wavfile
        print("SpeechT5 and Translation import successful")
    except ImportError as e:
        print(f"Error importing SpeechT5 or Translation: {e}")
        tts_installed = False
        translation_installed = False

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task='text-generation',
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,"max_length":"512"}
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])

# Text-to-Speech (TTS)
if tts_installed:
    try:
        # Initialize TTS pipeline
        tts = pipeline("text-to-speech", "microsoft/speecht5_tts")

        # Generate speech
        speech = tts(response["result"])[0].numpy() # corrected - access the numpy array directly

        # Save speech to file
        sampling_rate = tts.model.config.sample_rate
        scipy.io.wavfile.write("output.wav", rate=sampling_rate, data=speech)

        print("Speech generated and saved to output.wav")
    except Exception as e:
        print(f"Error generating speech: {e}")
else:
    print("TTS is not available because the SpeechT5 library failed to install.")

# Translation
if translation_installed:
    try:
        # Load translation pipeline
        translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr") #Translate from English to French
        translated_text = translator(response["result"])[0]['translation_text']
        print("Translated Text (French):", translated_text)
    except Exception as e:
        print(f"Error during translation: {e}")
else:
    print("Translation is not available because the necessary libraries failed to install.")