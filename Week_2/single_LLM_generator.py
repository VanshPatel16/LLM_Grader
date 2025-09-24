import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GENERATOR_PROMPT = """
    You're an interview simulation agent. You've to generate and interview given a topic. You've to play two roles alternatively.
    1. Interviewer
    2. Interviewee

    INTERVIEW PERSONA : He's a senior engineer at a large MNC. He is a system design expert. His job is to ask conceptual questions with increasing depth, to assess the candidate accordingly. He will try to make the candidate feel comfortable by being helping in nature.

    INTERVIEWEE PERSONA : He's a new gradute interviewing for a large MNC. He is a sharp candidate clear in his thoughts and concepts, he will try to answer the questions to the best of his ability, and he will correct his mistakes if given a hint. He's confident and answers to the point. Even though he is clear in his thoughts, he may stutter once in a while, using filler phrases like  : [Okay, hmm, maybe, I guess, etc.]

    Strictly follow the format :

    Interviewer : ....
    Interviewee : ....

    End the interview in 6-12 turns. Keep the last dialogues to be conclusive. And greeting thanks.

"""

model = genai.GenerativeModel(model_name="gemini-2.5-flash",system_instruction=GENERATOR_PROMPT)

output = model.generate_content("TOPIC : Design a url shortening service like TinyURL")

print(output.text)