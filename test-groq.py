from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load the API key from .env
load_dotenv()

# Connect to Groq
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Send our first message
response = llm.invoke(
    "You are the first AI component of RiskGuard-AI. "
    "Introduce yourself in one short sentence."
)

print(response.content)