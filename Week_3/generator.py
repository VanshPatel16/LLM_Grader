import google.generativeai as genai
from dotenv import load_dotenv
import os,sys,time
import yaml
import random

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# get prompts


def load_config(filepath="prompts.yaml"):
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
interviewee_prompt_templates = config.get("interviewee_prompt_templates",{})

def get_persona_prompts()-> tuple | None :
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

INITIAL_PROMPT = "TOPIC : Design a global live video streaming service like Youtube or Netflix"
MAX_TURNS = 20
STUDENT_TYPE = "good_student"
conversation_history = [{'role' : 'user' ,'parts' :INITIAL_PROMPT}]
interviewee_response_types = list(interviewee_prompt_templates.keys())
# print(interviewee_response_types)

def next_prompt(role: str,last: bool,question_difficulty="") -> str:
    if role == "Interviewer":
        if not last:
            next_prompt_template = prompt_templates.get("interviewer_reply_prompt",{}).get("template")
            return next_prompt_template
        else:
            concluding_template = prompt_templates.get("interviewer_conclude_prompt",{}).get("template")
            return concluding_template
    else:
        if not last:
            styles_config = config["student_personas"][STUDENT_TYPE][question_difficulty]
            style_weights = [style['weight'] for style in styles_config.values()]
            type_of_prompt = random.choices(interviewee_response_types,weights=style_weights,k=1)[0]
            next_prompt_template = config["interviewee_prompt_templates"][type_of_prompt]["template"]

            prompt = f"TYPE : {type_of_prompt}\n"
            prompt += next_prompt_template
            return prompt
        else:
            concluding_template = prompt_templates.get("interviewee_conclude_prompt",{}).get("template")

            return concluding_template
        

question_difficulty = "EASY"

filename = "result.txt"

with open(filename,'w',encoding='utf-8') as f:
    f.write(INITIAL_PROMPT)
    is_last = False
    for iter in range(2 * MAX_TURNS):

        is_last = is_last or (iter >= 2 * (MAX_TURNS - 1))

        prompt = ""
        response = ""
        response_txt = ""
        if iter % 2 == 0:
            # Interviewer's turn.
            prompt = next_prompt("Interviewer",is_last)
            history = conversation_history
            history.append({'role' : 'user', 'parts' : prompt})
            response = interviewer.generate_content(history)
            if "<EASY>" in response.text:
                question_difficulty = "EASY"
                response_txt = response.text.replace("<EASY>","").strip()
            elif "<MEDIUM>" in response.text:
                question_difficulty = "MEDIUM"
                response_txt = response.text.replace("<MEDIUM>","").strip()
            elif "<HARD>" in response.text:
                question_difficulty = "HARD"
                response_txt = response.text.replace("<HARD>","").strip()
            elif "<END_OF_INTERVIEW>" in response.text:
                response_txt = response.text.replace("<END_OF_INTERVIEW>","").strip()
                is_last = True
            elif not is_last:
                print("No difficulty assigned.")
                question_difficulty = "EASY"
                sys.exit(1)
            else:
                response_txt = response.text

        else:
            # Interviewee's turn.
            prompt = next_prompt("Interviewee",is_last,question_difficulty)
            history = conversation_history
            history.append({'role' : 'user', 'parts' : prompt})
            response = interviewee.generate_content(history)
            response_txt = response.text
        
        conversation_history.append({'role' : 'model','parts' : response_txt})
        f.write('\n' + response.text + '\n')
        print('\n' + response.text + '\n')
        if is_last and iter%2 != 0:
            break
        time.sleep(2)
    

