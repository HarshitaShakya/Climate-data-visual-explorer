import os
from google import genai

def init_gemini_client():
    api_key = os.getenv("AIzaSyBbvWrGHBaeNxxVFYVzOzG2mkuDMDP7TzU")  # Make sure to set this environment variable
    return genai.Client(api_key=api_key)

def generate_summary_with_gemini(content: str) -> str:
    client = init_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",  # Or "gemini-2.5-pro"
        contents=[{"content": content}]
    )
    return response[0].text if response else "Summary not available."

def generate_recommendations_with_gemini(content: str) -> str:
    prompt = (
        f"Analyze the following climate/weather data, including trends, anomalies, "
        f"and forecasts, and provide actionable recommendations for decision-makers:\n\n{content}"
    )
    return generate_summary_with_gemini(prompt)
