import google.generativeai as genai
from dotenv import load_dotenv
import os,sys
import yaml
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def load_config(filepath="single_LLM_gen_prompts.yaml"):
    """Loads YAML config file"""
    try:
        with open(filepath,'r') as f : 
            return yaml.safe_load(f)
        
    except FileNotFoundError : 
        print("File not found!")
        sys.exit(1)


config = load_config()
personas = config.get("personas",{})
prompt_templates = config.get("prompt_templates",{})

def get_prompt(topic: str) -> str | None:
    interviewer_persona = personas.get("interviewer",{}).get("description")
    interviewee_persona = personas.get("interviewee",{}).get("description")
    template = prompt_templates.get("generator_prompt",{}).get("template")

    return template.format(
        interviewer_persona=interviewer_persona,
        interviewee_persona=interviewee_persona,
        topic=topic
    )

TOPIC = "Design a URL shortening service like TinyURL"

GENERATOR_PROMPT = get_prompt(TOPIC)

model = genai.GenerativeModel(model_name="gemini-2.5-flash",system_instruction=GENERATOR_PROMPT)

output = model.generate_content(GENERATOR_PROMPT)

print(output.text)