import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from groq import Groq
from apify_client import ApifyClient
# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

apify_client=ApifyClient(os.getenv("APIFY_API_TOKEN"))

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF file.
    """
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def ask_groq(prompt, max_tokens=500):
    """
    Sends a prompt to the Groq API and returns the response.
    """
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # Groq supports LLaMA 3 models
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content

