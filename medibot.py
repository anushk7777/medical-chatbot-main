import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from gtts import gTTS
import speech_recognition as sr
import datetime
import json
import re
from PIL import Image
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
from disease_predictor import DiseasePredictor
from depression_assessment import PHQ9Assessment, VisualizationTools
from mental_health_tools import MentalHealthAssessment

# Load environment variables
load_dotenv(find_dotenv())
DB_FAISS_PATH = "vectorstore/db_faiss/"

# Set up page configuration
st.set_page_config(
    page_title="MediBot - Advanced Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-header {
        color: white;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .source-expander {
        margin-top: 1rem;
        border-top: 1px solid #4d5561;
        padding-top: 0.5rem;
    }
    .prediction-meter {
        margin-top: 1rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #2b313e;
    }
    .stProgress > div > div > div {
        background-color: #4299e1;
    }
    .high-risk .stProgress > div > div > div {
        background-color: #f56565;
    }
    .medium-risk .stProgress > div > div > div {
        background-color: #ed8936;
    }
    .low-risk .stProgress > div > div > div {
        background-color: #48bb78;
    }
    .btn-primary {
        background-color: #4299e1;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        border: none;
        cursor: pointer;
    }
    .btn-secondary {
        background-color: #718096;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        border: none;
        cursor: pointer;
    }
    .assessment-card {
        background-color: #2d3748;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #2b313e;
        border-radius: 0.25rem;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4299e1;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #a0aec0;
    }
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #4a5568;
        font-size: 0.75rem;
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


@st.cache_resource
def load_disease_predictor():
    return DiseasePredictor()


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task='text-generation',
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm


def text_to_speech(text):
    """Convert text to speech using gTTS and return audio bytes."""
    try:
        audio_bytes_io = io.BytesIO()
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)
        return audio_bytes_io.getvalue()
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None


def speech_to_text():
    """Record audio from microphone and convert to text."""
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening... Speak now.")
            audio = r.listen(source, timeout=5, phrase_time_limit=15)
            st.write("Processing your speech...")
            text = r.recognize_google(audio)
            return text
    except sr.RequestError:
        st.error("Could not request results from speech recognition service.")
        return None
    except sr.UnknownValueError:
        st.error("Could not understand audio. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error in speech recognition: {str(e)}")
        return None


def get_download_link(text, filename="MediBot_Chat_History.txt"):
    """Generate a download link for chat history."""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Chat History</a>'
    return href


def save_chat_history(messages):
    """Save chat history to a file."""
    chat_text = "MediBot Chat History\n" + "=" * 50 + "\n"
    chat_text += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for msg in messages:
        if msg['role'] == 'user':
            chat_text += f"You: {msg['content']}\n\n"
        else:
            # Remove source docs from assistant messages for cleaner export
            content = msg['content']
            if "Source Docs:" in content:
                content = content.split("Source Docs:")[0]
            chat_text += f"MediBot: {content.strip()}\n\n"

    return chat_text


def main():
    # Initialize disease predictor and assessment tools
    disease_predictor = load_disease_predictor()
    phq9_assessment = PHQ9Assessment()
    mental_health_tools = MentalHealthAssessment()
    visualization_tools = VisualizationTools()

    # Sidebar layout and content
    st.sidebar.title("🩺 MediBot")
    st.sidebar.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=100)
    st.sidebar.info(
        "MediBot is your advanced medical assistant. Ask questions, get assessments, and receive health insights."
    )

    # App mode selection
    app_mode = st.sidebar.radio("Select Mode:", [
        "💬 Chat with MediBot",
        "🔍 Disease Predictor",
        "📊 Depression Assessment",
        "📈 Mental Health Tools",
        "⚙️ Settings"
    ])

    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'tts_enabled' not in st.session_state:
        st.session_state.tts_enabled = False

    if 'phq9_score' not in st.session_state:
        st.session_state.phq9_score = None

    if 'phq9_history' not in st.session_state:
        st.session_state.phq9_history = []

    if 'voice_input_enabled' not in st.session_state:
        st.session_state.voice_input_enabled = False

    # Chat with MediBot mode
    if app_mode == "💬 Chat with MediBot":
        st.title("🩺 MediBot: Your Advanced Medical Assistant")

        # Display chat messages
        for message in st.session_state.messages:
            if message['role'] == 'user':
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message user">
                        <div class="chat-header">You</div>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                with st.container():
                    # Split content into answer and sources
                    content = message['content']
                    if "Source Docs:" in content:
                        answer, sources_text = content.split("Source Docs:", 1)

                        st.markdown(f"""
                        <div class="chat-message assistant">
                            <div class="chat-header">MediBot</div>
                            {answer.strip()}
                        </div>
                        """, unsafe_allow_html=True)

                        # Sources expander
                        with st.expander("View Sources"):
                            try:
                                import re
                                source_pattern = r"source='([^']*)'.*?page=(\d+|'N/A')"
                                source_matches = re.findall(source_pattern, sources_text)

                                if source_matches:
                                    source_data = []
                                    for source, page in source_matches:
                                        source_data.append({
                                            "Source": source,
                                            "Page": page if page != "'N/A'" else "N/A"
                                        })

                                    st.dataframe(source_data)
                                else:
                                    st.write("Source information available but could not be parsed.")
                                    st.text(sources_text[:500] + "..." if len(sources_text) > 500 else sources_text)
                            except Exception as e:
                                st.write(f"Error displaying sources: {str(e)}")

                        # Text-to-speech for the answer only
                        if st.session_state.tts_enabled:
                            audio_bytes = text_to_speech(answer.strip())
                            if audio_bytes:
                                st.audio(audio_bytes, format="audio/mp3")
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant">
                            <div class="chat-header">MediBot</div>
                            {content}
                        </div>
                        """, unsafe_allow_html=True)

        # Voice input button
        if st.session_state.voice_input_enabled:
            if st.button("🎤 Speak to MediBot"):
                voice_text = speech_to_text()
                if voice_text:
                    prompt = voice_text
                    st.session_state.messages.append({'role': 'user', 'content': prompt})
                    with st.spinner("Thinking..."):
                        process_user_query(prompt)

        # Text input for user query
        prompt = st.chat_input("Ask your medical question...")

        if prompt:
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            with st.spinner("Thinking..."):
                process_user_query(prompt)

        # Chat controls
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.session_state.messages:
                chat_text = save_chat_history(st.session_state.messages)
                st.markdown(get_download_link(chat_text), unsafe_allow_html=True)

    # Disease Predictor mode
    elif app_mode == "🔍 Disease Predictor":
        st.title("🔍 Disease Prediction Based on Symptoms")
        st.markdown("""
        Describe your symptoms in detail below, and I'll predict potential conditions 
        based on pattern recognition. Please note this is not a substitute for professional medical diagnosis.
        """)

        # Input for symptoms
        symptoms_text = st.text_area("Describe your symptoms:",
                                     placeholder="Example: I have high fever, cough, and difficulty breathing for the past 3 days.",
                                     height=150)

        col1, col2 = st.columns([3, 1])
        with col1:
            predict_button = st.button("Predict Possible Conditions")
        with col2:
            if st.session_state.voice_input_enabled:
                if st.button("🎤 Speak Symptoms"):
                    voice_text = speech_to_text()
                    if voice_text:
                        symptoms_text = voice_text
                        st.experimental_rerun()

        if predict_button and symptoms_text:
            with st.spinner("Analyzing symptoms..."):
                try:
                    # Get predictions
                    predictions = disease_predictor.get_top_predictions(symptoms_text, top_n=5)

                    # Display results
                    st.markdown("### Prediction Results")
                    st.markdown("Based on your symptoms, here are the most likely conditions:")

                    # Create a container for the meters
                    prediction_container = st.container()

                    with prediction_container:
                        for disease, probability in predictions:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                # Assign color class based on probability
                                risk_class = ""
                                if probability > 0.7:
                                    risk_class = "high-risk"
                                elif probability > 0.4:
                                    risk_class = "medium-risk"
                                else:
                                    risk_class = "low-risk"

                                st.markdown(f"<div class='{risk_class}'>", unsafe_allow_html=True)
                                st.progress(probability)
                                st.markdown("</div>", unsafe_allow_html=True)
                            with col2:
                                st.metric(label=disease, value=f"{probability * 100:.1f}%")

                        # Add to chat history for reference
                        result_text = f"Based on symptoms: '{symptoms_text}', the following conditions were predicted:\n\n"
                        for disease, probability in predictions:
                            result_text += f"- {disease}: {probability * 100:.1f}%\n"

                        st.session_state.messages.append({
                            'role': 'user',
                            'content': f"I have the following symptoms: {symptoms_text}"
                        })
                        st.session_state.messages.append({
                            'role': 'assistant',
                            'content': result_text
                        })

                    # Display disclaimer
                    st.markdown("""
                    ---
                    **Disclaimer:** This prediction is based on pattern recognition and should not be 
                    considered as medical advice. Please consult with a healthcare professional for proper 
                    diagnosis and treatment.
                    """)

                    # Suggest follow-up actions
                    st.info("💡 Switch to 'Chat with MediBot' mode to ask detailed questions about these conditions.")

                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

    # Depression Assessment Mode
    elif app_mode == "📊 Depression Assessment":
        st.title("📊 Depression Assessment Tool")
        st.markdown("""
        This tool uses the PHQ-9 (Patient Health Questionnaire), a validated screening tool for depression. 
        Answer honestly about how you've been feeling over the past two weeks.
        """)

        # Display PHQ-9 questionnaire
        phq9_score = phq9_assessment.display_questionnaire()

        # If score is calculated, save and display results
        if phq9_score is not None:
            # Save score with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
            if len(st.session_state.phq9_history) > 0:
                # Check if we already have an entry for today
                if st.session_state.phq9_history[-1][0] == timestamp:
                    st.session_state.phq9_history[-1] = (timestamp, phq9_score)
                else:
                    st.session_state.phq9_history.append((timestamp, phq9_score))
            else:
                st.session_state.phq9_history.append((timestamp, phq9_score))

            st.session_state.phq9_score = phq9_score

            # Display score and interpretation
            severity, interpretation = phq9_assessment.interpret_score(phq9_score)

            # Visualization of the score
            st.markdown("### Your PHQ-9 Score")

            # Create score gauge
            score_gauge = visualization_tools.create_phq9_gauge(phq9_score)
            st.plotly_chart(score_gauge, use_container_width=True)

            st.markdown(f"**Severity: {severity}**")
            st.markdown(f"**Interpretation:** {interpretation}")

            # Display history if available
            if len(st.session_state.phq9_history) > 1:
                st.markdown("### Your Depression Score History")
                history_chart = visualization_tools.create_phq9_history_chart(st.session_state.phq9_history)
                st.plotly_chart(history_chart, use_container_width=True)

                # Show trend analysis
                trend = visualization_tools.analyze_trend(st.session_state.phq9_history)
                st.markdown(f"**Trend Analysis:** {trend}")

            # Add recommendations based on score
            st.markdown("### Recommendations")
            recommendations = phq9_assessment.get_recommendations(phq9_score)

            for rec in recommendations:
                st.markdown(f"- {rec}")

            # Medical disclaimer
            st.warning("""
            **Medical Disclaimer:** This assessment is not a substitute for professional medical advice, 
            diagnosis, or treatment. Always seek the advice of your physician or other qualified health 
            provider with any questions you may have regarding a medical condition.
            """)

    # Mental Health Tools mode
    elif app_mode == "📈 Mental Health Tools":
        st.title("📈 Mental Health Insights and Tools")

        tool_choice = st.radio("Select Tool:", [
            "Depression Course Visualization",
            "Treatment Effectiveness Comparison",
            "Risk Factor Analysis"
        ])

        if tool_choice == "Depression Course Visualization":
            st.markdown("### Depression Course Patterns")
            st.markdown("""
            This visualization shows typical depression trajectories based on clinical research.
            Understanding these patterns can help recognize and manage depression more effectively.
            """)

            pattern_type = st.selectbox(
                "Select depression pattern to visualize:",
                ["Single Episode with Recovery", "Recurrent Depression", "Chronic Depression", "Treatment Response"]
            )

            chart = mental_health_tools.create_depression_course_chart(pattern_type)
            st.plotly_chart(chart, use_container_width=True)

            # Description of the pattern
            st.markdown("#### Pattern Explanation")
            st.markdown(mental_health_tools.get_pattern_description(pattern_type))

        elif tool_choice == "Treatment Effectiveness Comparison":
            st.markdown("### Treatment Effectiveness Comparison")
            st.markdown("""
            This tool compares the effectiveness of different depression treatments based on
            research data. The chart shows typical response rates from clinical studies.
            """)

            depression_type = st.selectbox(
                "Select depression type:",
                ["Major Depression", "Treatment-Resistant Depression", "Depression with Anxiety", "Bipolar Depression"]
            )

            chart = mental_health_tools.create_treatment_comparison_chart(depression_type)
            st.plotly_chart(chart, use_container_width=True)

            # Treatment recommendations
            st.markdown("#### Evidence-Based Considerations")
            st.markdown(mental_health_tools.get_treatment_recommendations(depression_type))

        elif tool_choice == "Risk Factor Analysis":
            st.markdown("### Depression Risk Factor Analysis")
            st.markdown("""
            This tool allows you to analyze how different risk factors contribute to depression.
            The visualization is based on epidemiological research data.
            """)

            # Risk factor inputs
            col1, col2 = st.columns(2)

            with col1:
                gender = st.radio("Gender", ["Female", "Male"])
                age_group = st.selectbox("Age Group", ["18-29", "30-44", "45-59", "60+"])
                family_history = st.radio("Family History of Depression", ["Yes", "No"])

            with col2:
                chronic_illness = st.radio("Chronic Physical Illness", ["Yes", "No"])
                recent_trauma = st.radio("Recent Stressful Life Event", ["Yes", "No"])
                social_support = st.selectbox("Social Support Level", ["Low", "Medium", "High"])

            # Calculate risk based on selections
            risk_score, factor_weights = mental_health_tools.calculate_risk_score(
                gender, age_group, family_history, chronic_illness, recent_trauma, social_support
            )

            # Display risk score
            st.markdown("### Your Risk Analysis")
            risk_gauge = visualization_tools.create_risk_gauge(risk_score)
            st.plotly_chart(risk_gauge, use_container_width=True)

            # Display factor breakdown
            st.markdown("### Risk Factor Breakdown")
            factor_chart = visualization_tools.create_factor_chart(factor_weights)
            st.plotly_chart(factor_chart, use_container_width=True)

            # Recommendations based on risk factors
            st.markdown("### Personalized Recommendations")
            recommendations = mental_health_tools.get_risk_recommendations(
                risk_score, gender, age_group, family_history, chronic_illness, recent_trauma, social_support
            )

            for rec in recommendations:
                st.markdown(f"- {rec}")

    # Settings mode
    elif app_mode == "⚙️ Settings":
        st.title("⚙️ Settings")
        st.markdown("Configure your MediBot experience")

        # Interface settings
        st.markdown("### Interface Settings")
        st.session_state.tts_enabled = st.checkbox("Enable Text-to-Speech", value=st.session_state.tts_enabled)
        st.session_state.voice_input_enabled = st.checkbox("Enable Voice Input",
                                                           value=st.session_state.voice_input_enabled)

        # Accessibility settings
        st.markdown("### Data & Privacy")
        if st.button("Delete All Chat History"):
            st.session_state.messages = []
            st.success("Chat history has been deleted!")

        if st.button("Reset All Assessment Data"):
            st.session_state.phq9_score = None
            st.session_state.phq9_history = []
            st.success("All assessment data has been reset!")

        # About section
        st.markdown("### About MediBot")
        st.markdown("""
        **Version:** 2.0

        **Features:**
        - Medical Q&A with PDF knowledge base
        - Disease prediction based on symptoms
        - Depression assessment and tracking
        - Mental health visualization tools
        - Voice interaction capabilities

        **Disclaimer:** MediBot is not a substitute for professional medical advice,
        diagnosis, or treatment. Always seek the advice of your physician or other
        qualified health provider with any questions you may have regarding a medical condition.
        """)

    # Footer
    st.markdown("""
    <div class="footer">
        MediBot Advanced Medical Assistant © 2025
    </div>
    """, unsafe_allow_html=True)


def process_user_query(prompt):
    """Process the user query and get response from the LLM"""

    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer.
    Dont provide anything out of the given context.
    Context: {context}
    Question: {question}
    Start the answer directly. No small talk please.
    """

    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the vector store")
            return

        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        response = qa_chain.invoke({'query': prompt})
        result = response["result"]
        source_documents = response["source_documents"]

        # Format source documents for display
        result_to_show = result + "\nSource Docs:" + str(source_documents)

        # Display assistant message with the latest message
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="chat-header">MediBot</div>
            {result.strip()}
        </div>
        """, unsafe_allow_html=True)

        # Create sources expander
        with st.expander("View Sources"):
            if source_documents:
                # Format source data for display
                source_data = []
                for i, doc in enumerate(source_documents):
                    source_data.append({
                        "Source": doc.metadata.get("source", "Unknown"),
                        "Page": doc.metadata.get("page", "N/A"),
                        "Excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })
                st.dataframe(source_data)
            else:
                st.write("No source documents available")

        # Add to session state
        st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        # Text-to-speech for response
        if st.session_state.tts_enabled:
            audio_bytes = text_to_speech(result.strip())
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
