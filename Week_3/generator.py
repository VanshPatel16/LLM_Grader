import google.generativeai as genai
from dotenv import load_dotenv
import os,sys,time
import yaml
import random
import json
from datetime import datetime

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
conversation_history = [{'role' : 'user' ,'parts' :INITIAL_PROMPT}]
interviewee_response_types = list(interviewee_prompt_templates.keys())
# print(interviewee_response_types)

def next_prompt(role: str, last: bool, question_difficulty="", student_type="poor_student") -> str:
    if role == "Interviewer":
        next_prompt_template = prompt_templates.get("interviewer_reply_prompt",{}).get("template")
        return next_prompt_template
    else:
        styles_config = config["student_personas"][student_type][question_difficulty]
        style_weights = [style['weight'] for style in styles_config.values()]
        type_of_prompt = random.choices(interviewee_response_types,weights=style_weights,k=1)[0]
        next_prompt_template = config["interviewee_prompt_templates"][type_of_prompt]["template"]

        prompt = f"TYPE : {type_of_prompt}\n"
        prompt += next_prompt_template
        return prompt
        

def generate_conversation(conversation_id, student_type=None, topic=None, max_turns=None):
    """Generate a single conversation and return it as a dictionary"""
    topics_pool = [
        "Design a global live video streaming service like Youtube or Netflix",
        "Design a ride-sharing service like Uber or Lyft",
        "Design a real-time messaging system like WhatsApp or Slack",
        "Design a URL shortening service like bit.ly",
        "Design an e-commerce platform like Amazon",
        "Design a large-scale logging and metrics system",
        "Design a scalable job scheduler and worker system",
        "Design a distributed file storage system like Google Drive",
        "Design a recommendation engine for content platforms",
        "Design a real-time analytics dashboard system",
        "Design a multi-tenant SaaS application platform"
    ]
    if topic is None:
        topic = random.choice(topics_pool)
    if student_type is None:
        student_type = random.choice(["poor_student", "average_student", "good_student"])
    if max_turns is None:
        max_turns = random.randint(3, 6)
    
    # Reset conversation state
    conversation_history = [{'role' : 'user' ,'parts' : topic}]
    question_difficulty = "EASY"
    turns = []
    difficulty_counts = {"EASY": 0, "MEDIUM": 0, "HARD": 0}
    response_type_counts = {"clear": 0, "confused": 0, "misunderstood": 0, "wrong": 0}
    
    
    is_last = False
    turn_number = 0
    
    for exchange_index in range(max_turns):
        # Interviewer's turn
        prompt = next_prompt("Interviewer", is_last, student_type=student_type)
        conversation_history.append({'role' : 'user', 'parts' : prompt})
        response = interviewer.generate_content(conversation_history)
        response_txt = response.text
        current_difficulty = None
        if "<EASY>" in response_txt:
            question_difficulty = "EASY"
            current_difficulty = "EASY"
            response_txt = response_txt.replace("<EASY>","").strip()
            difficulty_counts["EASY"] += 1
        elif "<MEDIUM>" in response_txt:
            question_difficulty = "MEDIUM"
            current_difficulty = "MEDIUM"
            response_txt = response_txt.replace("<MEDIUM>","").strip()
            difficulty_counts["MEDIUM"] += 1
        elif "<HARD>" in response_txt:
            question_difficulty = "HARD"
            current_difficulty = "HARD"
            response_txt = response_txt.replace("<HARD>","").strip()
            difficulty_counts["HARD"] += 1
        elif "<END_OF_INTERVIEW>" in response_txt:
            is_last = True
            response_txt = response_txt.replace("<END_OF_INTERVIEW>","").strip()
        
        if response_txt.startswith("Interviewer:"):
            response_txt = response_txt[12:].strip()
        
        turn_number += 1
        turns.append({
            "turn_number": turn_number,
            "speaker": "interviewer",
            "content": response_txt,
            "difficulty": current_difficulty,
            "response_type": None
        })
        conversation_history.append({'role' : 'model','parts' : response_txt})
        print(f"Turn {turn_number} (Interviewer): {response_txt[:100]}...")
        time.sleep(2)

        # Interviewee's turn (always follow, even if interviewer ended)
        prompt = next_prompt("Interviewee", is_last, question_difficulty, student_type)
        conversation_history.append({'role' : 'user', 'parts' : prompt})
        response = interviewee.generate_content(conversation_history)
        response_txt = response.text
        current_response_type = None
        if "TYPE :" in prompt:
            current_response_type = prompt.split("TYPE :")[1].split("\n")[0].strip()
            response_type_counts[current_response_type] += 1
        if "Interviewee[" in response_txt:
            response_txt = response_txt.split("]:", 1)[1].strip()
        elif response_txt.startswith("Interviewee:"):
            response_txt = response_txt[12:].strip()
        
        turn_number += 1
        turns.append({
            "turn_number": turn_number,
            "speaker": "interviewee",
            "content": response_txt,
            "difficulty": None,
            "response_type": current_response_type
        })
        conversation_history.append({'role' : 'model','parts' : response_txt})
        print(f"Turn {turn_number} (Interviewee): {response_txt[:100]}...")
        time.sleep(2)

        # If interviewer signaled end, finish after interviewee reply
        if is_last:
            break
    
    # Calculate most frequently occurring difficulty
    most_frequent_difficulty = max(difficulty_counts.items(), key=lambda x: x[1])[0] if any(difficulty_counts.values()) else "Easy"
    
    # Create ordered conversation list with {speaker, speech}
    conversation_list = []
    for turn in turns:
        conversation_list.append({
            "speaker": turn["speaker"],
            "speech": turn["content"]
        })
    
    # Calculate exchanges (pairs of interviewer-interviewee turns)
    exchanges = len([t for t in turns if t["speaker"] == "interviewee"])
    
    conversation = {
        "id": int(conversation_id) if isinstance(conversation_id, int) else int(conversation_id.split("_")[1]),
        "student_level": student_type,
        "difficulty": most_frequent_difficulty.title(),  # Capitalize first letter
        "exchanges": exchanges,
        "topic": topic.replace("TOPIC : ", ""),  # Remove "TOPIC : " prefix
        "conversation": conversation_list,
        "difficulty_distribution": difficulty_counts,
        "reply_distribution": response_type_counts
    }
    
    return conversation

def generate_dataset(num_conversations=5, student_types=None, topics=None, output_file="interview_dataset.json"):
    """Generate a dataset of multiple conversations with API limit handling.
    Appends to output_file if it exists (does not overwrite)."""
    if student_types is None:
        student_types = ["poor_student", "average_student", "good_student"]
    if topics is None:
        # Use the same topics pool as generate_conversation
        topics = [
            "Design a global live video streaming service like Youtube or Netflix",
            "Design a ride-sharing service like Uber or Lyft",
            "Design a real-time messaging system like WhatsApp or Slack",
            "Design a URL shortening service like bit.ly",
            "Design an e-commerce platform like Amazon",
            "Design a large-scale logging and metrics system",
            "Design a scalable job scheduler and worker system",
            "Design a distributed file storage system like Google Drive",
            "Design a recommendation engine for content platforms",
            "Design a real-time analytics dashboard system",
            "Design a multi-tenant SaaS application platform"
        ]
    
    # Load existing dataset if present to append
    conversations = []
    existing_student_types = set()
    existing_topics = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            if isinstance(existing, dict) and 'conversations' in existing:
                conversations = existing['conversations']
                di = existing.get('dataset_info', {})
                for c in conversations:
                    existing_student_types.add(c.get('student_level'))
                    existing_topics.add(c.get('topic'))
                print(f"Loaded existing dataset with {len(conversations)} conversations from {output_file}")
    except FileNotFoundError:
        pass
    
    for i in range(num_conversations):
        next_id = (max((c['id'] for c in conversations), default=0) + 1)
        conversation_id = f"conversation_{next_id:03d}"
        student_type = random.choice(student_types)
        topic = random.choice(topics)
        
        print(f"\nGenerating conversation {i+1}/{num_conversations} (ID: {conversation_id})")
        print(f"Student type: {student_type}, Topic: {topic}")
        
        try:
            conversation = generate_conversation(conversation_id, student_type, topic, None)
            conversations.append(conversation)
            print(f"Completed conversation {i+1} with {conversation['exchanges']} exchanges")
            
            # Save progress after each conversation
            dataset = {
                "dataset_info": {
                    "total_conversations": len(conversations),
                    "generated_at": datetime.now().isoformat(),
                    "student_types_used": list(set([c["student_level"] for c in conversations]) | existing_student_types),
                    "topics_covered": list(set([c["topic"] for c in conversations]) | existing_topics),
                    "status": f"Generated {len(conversations)}/{num_conversations} conversations"
                },
                "conversations": conversations
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"Progress saved to {output_file}")
            
        except Exception as e:
            print(f"Error generating conversation {i+1}: {str(e)}")
            print(f"API limit or error encountered. Saving {len(conversations)} completed conversations.")
            break
    
    return conversations

# Generate a single conversation for testing
if __name__ == "__main__":
    # Generate single conversation
    conversation = generate_conversation(1, None, None, None)
    
    # Save to JSON file
    filename = "conversation_dataset.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)
    
    print(f"\nConversation saved to {filename}")
    print(f"Total exchanges: {conversation['exchanges']}")
    print(f"Student level: {conversation['student_level']}")
    print(f"Most frequent difficulty: {conversation['difficulty']}")
    print(f"Difficulty distribution: {conversation['difficulty_distribution']}")
    print(f"Reply distribution: {conversation['reply_distribution']}")
    

