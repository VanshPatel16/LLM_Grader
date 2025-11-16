import json
import os
from openai import OpenAI
from datasets import Dataset
from dotenv import load_dotenv
import random, time

# Configure OpenAI API
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === CONFIG ===
INPUT_FILE = "interview_dataset.json"
OUTPUT_FILE = "interview_labels.json"
MODEL_NAME = "gpt-5-nano" 

# === PROMPT TEMPLATE ===
def build_prompt(conversation):

    topic = conversation["topic"]
    convo = ""
    for exchange in conversation["conversation"] : 
        if exchange["speaker"] == "interviewer" :
            convo += ("Interviewer : " + exchange["speech"])
        else :
            convo += ("Interviewee : " + exchange["speech"])
        convo += "\n"
    prompt = f"""
        You are an expert technical interviewer and evaluator.  
    Your task is to **evaluate the candidate's performance** in the following system design interview conversation.

    You will use the **10 detailed rubric criteria** below.  
    For each criterion, give a score between **0 and 2** based on the evidence from the conversation.

    ---

    ### Rubric Criteria

    1. **Problem Understanding & Requirement Gathering** – How well the candidate clarifies scope, asks relevant questions, and identifies key constraints.  
    *Evidence:* Questions about scale, users, features, edge cases, quantifying targets (QPS, storage, users).  

    2. **Structured Problem-Solving Approach** – Ability to break down the problem logically and communicate their thought process clearly.  
    *Evidence:* Organized thinking, logical progression, building incrementally.  

    3. **High-Level Architecture & Design Evolution** – How the candidate develops and refines their system architecture.  
    *Evidence:* Component identification, service boundaries, data flow explanations, refining architecture with feedback.  

    4. **Technical Depth & Implementation Details** – Ability to dive deeper into technical aspects when prompted.  
    *Evidence:* Concrete design choices — database selection, APIs, algorithms, data structures, technology justification.  

    5. **Scalability & Performance Reasoning** – Handling of large-scale, high-performance challenges.  
    *Evidence:* Scaling strategies, identifying bottlenecks, caching, partitioning, replication approaches.  

    6. **Trade-off Analysis & Decision Justification** – Ability to reason about alternatives and justify decisions.  
    *Evidence:* Discussion of pros/cons, CAP theorem tradeoffs, cost vs performance tradeoffs, justification for choices.  

    7. **Handling Follow-up Questions & Adaptability** – Responsiveness to interviewer feedback and ability to evolve the design.  
    *Evidence:* Adapting designs, thoughtful responses to edge cases, handling challenges, acknowledging unknowns.  

    8. **Reliability & Fault Tolerance Considerations** – Awareness of resilience and fault tolerance strategies.  
    *Evidence:* Mentions of redundancy, replication, circuit breakers, disaster recovery, handling failures.  

    9. **Communication & Collaboration** – Clarity, tone, and engagement style in interacting with the interviewer.  
    *Evidence:* Clear articulation, good examples, asking for clarifications, collaborative attitude.  

    10. **Completeness & Time Tracking** – How well the candidate covers major areas of the design within time.  
        *Evidence:* Addressing key aspects, maintaining progress, not getting stuck on details.

    ---

    ### Scoring Scale
    0 → Not addressed or fundamentally incorrect  
    1 → Basic response with some correct elements  
    2 → Excellent, detailed, technically sound response  

    ---

    ### Input Conversation

    {convo}

    ---

    ### Output Format (strictly JSON)

    Return **only** the following JSON, with integer scores from 0–2 for each rubric category:

    {{
    "problem_understanding": int,
    "structured_approach": int,
    "architecture_evolution": int,
    "technical_depth": int,
    "scalability_reasoning": int,
    "tradeoff_analysis": int,
    "adaptability": int,
    "reliability": int,
    "communication": int,
    "completeness": int
    }}

    Important: Do not give all scores as 2 very easily, give only if the candidate is perfect.
    Reflect realistic performance variation — some criteria should score 0 or 1 when appropriate.

    Do not include explanations, comments, or reasoning.  
    Return valid JSON **only**.
    """
    return prompt.strip()


def label_dataset():
    # Extract the "conversations" field first
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)["conversations"]

    # Then load into a Dataset object
    dataset = Dataset.from_list(data)
    
    results = []
    count = 0
    samples = [sample for sample in dataset]
    
    for sample in samples:
        count += 1
        prompt = build_prompt(sample)
        print(f"Labelling entry no. : {count}, id no. : {sample['id']}")
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            # parse JSON safely
            text = response.choices[0].message.content.strip()
            try:
                label = json.loads(text)
            except json.JSONDecodeError:
                # handle model returning extra text or invalid JSON
                start = text.find("{")
                end = text.rfind("}")
                label = json.loads(text[start:end+1])

            results.append({
                "id": sample["id"],
                "label": label
            })

            time.sleep(2)  # Reduced wait time for GPT API (adjust as needed for rate limits)

        except Exception as e:
            print(f"Error on sample: {e}")
            continue

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved labeled dataset to {OUTPUT_FILE}")
    print(count)


if __name__ == "__main__":
    label_dataset()
