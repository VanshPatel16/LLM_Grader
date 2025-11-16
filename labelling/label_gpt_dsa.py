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
INPUT_FILE = "dsa_dataset.json"
OUTPUT_FILE = "dsa_labels.json"
MODEL_NAME = "gpt-5-nano"  # or gpt-3.5-turbo for faster/cheaper results

# === PROMPT TEMPLATE ===
def build_prompt(conversation):
    """
    Build a prompt for evaluating DSA interview performance.
    """
    convo = ""
    for exchange in conversation:
        speaker = exchange["speaker"]
        speech = exchange["speech"]
        
        if speaker == "interviewer":
            convo += f"Interviewer: {speech}\n\n"
        else:
            convo += f"Interviewee: {speech}\n\n"
    
    prompt = f"""
You are an expert technical interviewer and evaluator specializing in Data Structures and Algorithms (DSA) problems.

Your task is to **evaluate the interviewee's performance** in the following DSA problem-solving interview conversation.

You will use the **8 detailed rubric criteria** below.  
For each criterion, give a score between **0 and 2** based on the evidence from the conversation.

---

### Rubric Criteria

1. **Ask Clarifying Questions** – How well the interviewee clarifies the problem, asks about constraints, edge cases, and requirements.  
   *Evidence:* Questions about input size, data types, range of values, special cases, or problem scope.  

2. **Propose Brute Force Solution** – Ability to quickly outline a straightforward, brute-force approach to the problem.  
   *Evidence:* Immediately providing a naive/brute-force solution, even if not optimal, showing basic problem comprehension.  

3. **Provide Space + Time Complexity** – Ability to analyze and articulate time and space complexity clearly.  
   *Evidence:* Mentioning Big-O notation, explaining how complexity scales, comparing brute-force vs optimized complexity.  

4. **Reach Optimal Solution Finally** – Whether the interviewee eventually arrives at an optimized or near-optimal solution.  
   *Evidence:* Improving upon initial approach, using better data structures/algorithms, reaching O(n log n) or better where applicable.  

5. **Handle Edge Cases** – Awareness and handling of edge cases and boundary conditions.  
   *Evidence:* Discussing null inputs, empty arrays, single elements, duplicates, negative numbers, or problem-specific edge cases.  

6. **Correct Explanation of Approach** – Clarity and correctness in explaining the algorithm/approach.  
   *Evidence:* Clear step-by-step explanation, correct algorithm logic, proper use of technical terminology.  

7. **Polite + Respectful Tone** – Professional communication style and respectful engagement with the interviewer.  
   *Evidence:* Courteous language, listening to feedback, not being defensive, positive attitude throughout.  

8. **Logical Progression of Conversation** – How well the interview flows logically from problem understanding to solution.  
   *Evidence:* Building incrementally on ideas, following the interviewer's guidance, organized thought process.  

---

### Scoring Scale
0 → Not addressed or fundamentally incorrect / Absent  
1 → Basic or partial response with some correct elements  
2 → Excellent, detailed, complete, and technically sound response  

---

### Interview Conversation

{convo}

---

### Output Format (strictly JSON)

Return **only** the following JSON, with integer scores from 0–2 for each rubric category:

{{
    "ask_clarifying_questions": int,
    "propose_brute_force": int,
    "space_time_complexity": int,
    "reach_optimal_solution": int,
    "handle_edge_cases": int,
    "correct_explanation": int,
    "polite_respectful_tone": int,
    "logical_progression": int
}}

Important: 
- Give scores realistically based on actual performance
- Do not give all scores as 2 unless the interviewee demonstrated excellence in all areas
- A score of 1 is appropriate when the interviewee showed basic understanding but lacked depth or completeness
- A score of 0 is appropriate when the criterion was not addressed or the response was incorrect

Do not include explanations, comments, or reasoning.  
Return valid JSON **only**.
"""
    return prompt.strip()


def label_dataset():
    # Load the cleaned DSA dataset
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    count = 0
    
    for interview in data:
        count += 1
        conversation = interview.get("conversation", [])
        interview_id = interview.get("id")
        
        prompt = build_prompt(conversation)
        print(f"Labelling entry no. : {count}, id no. : {interview_id}")
        
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
            
            # Parse JSON safely
            text = response.choices[0].message.content.strip()
            try:
                label = json.loads(text)
            except json.JSONDecodeError:
                # Handle model returning extra text or invalid JSON
                start = text.find("{")
                end = text.rfind("}")
                label = json.loads(text[start:end+1])

            results.append({
                "id": interview_id,
                "label": label
            })

            # time.sleep(2)  # Rate limiting

        except Exception as e:
            print(f"Error on interview {interview_id}: {e}")
            continue

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved labeled DSA dataset to {OUTPUT_FILE}")
    print(f"Total interviews labeled: {len(results)}/{len(data)}")


if __name__ == "__main__":
    label_dataset()
