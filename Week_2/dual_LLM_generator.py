import google.generativeai as genai
from dotenv import load_dotenv
import os,sys,time
import yaml
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# get prompts

def load_config(filepath="dual_LLM_gen_prompts.yaml"):
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

def get_persona_prompts()->tuple | None :
    """Loads persona prompts from YAML file"""
    interviewer_persona = personas.get("interviewer",{}).get("description")
    interviewee_persona = personas.get("interviewee",{}).get("description")

    if interviewee_persona and interviewer_persona:
        return (interviewer_persona,interviewee_persona)
    else:
        return None
        

prompt_pair = get_persona_prompts()

if prompt_pair:
    INTERVIEWER_PERSONA,INTERVIEWEE_PERSONA = prompt_pair
else:
    print("Could not retrieve personas.")
    sys.exit(1)

interviewer = genai.GenerativeModel(model_name="gemini-2.5-flash",system_instruction=INTERVIEWER_PERSONA)

interviewee = genai.GenerativeModel(model_name="gemini-2.5-flash",system_instruction=INTERVIEWEE_PERSONA)

INITIAL_PROMPT = "TOPIC : Design a url shortening service like TinyURL"
MAX_TURNS = 4
conversation_history = [INITIAL_PROMPT]


def next_prompt(role: str,last: bool) -> str:
    if role == "Interviewer":
        if not last:
            next_prompt_template = prompt_templates.get("reply_prompt",{}).get("template")
            return next_prompt_template.format(
                role=role,
            )
        else:
            concluding_template = prompt_templates.get("conclude_prompt",{}).get("template")
            return concluding_template
    else:
        
        next_prompt_template = prompt_templates.get("reply_prompt",{}).get("template")
        return next_prompt_template.format(
            role=role,
        )
        

for iter in range(2 * MAX_TURNS):
    is_last = (iter == 2 * (MAX_TURNS - 1))

    prompt = ""
    response = ""
    if iter % 2 == 0:
        # Interviewer's turn.
        prompt = next_prompt("Interviewer",is_last)
        history = conversation_history
        history.append(prompt)
        response = interviewer.generate_content(history)

    else:
        # Interviewee's turn.
        prompt = next_prompt("Interviewee",is_last)
        history = conversation_history
        history.append(prompt)
        response = interviewee.generate_content(history)
    
    conversation_history.append(response.text)
    # print('\nPrompt: ' + prompt + '\n')
    print('\n' + response.text + '\n')
    time.sleep(1)
    

