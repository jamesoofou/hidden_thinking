import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def call_llm(messages, model="gpt-4o-2024-08-06", temperature=1):
    response = await client.responses.create(
        model=model,
        input=messages,
        store=False,
        temperature=temperature,
    )
    return response.output_text

def to_dict(role, content):
    return {"role": role, "content": content}

async def run_single_attempt(messages, attempt_num, correct_answer):
    try:
        llm_answer = int(await call_llm(messages))
        print(f'attempt {attempt_num}: {llm_answer}')
        return llm_answer == correct_answer
    except Exception as e:
        print(f'Error in attempt {attempt_num}: {e}')
        return False

async def main():
    prompt_content = 'Please respond to the following question with just the numeric answer, nothing else. What is 382 * 4837?'
    correct_answer = 1847734
    prompt = to_dict('user', prompt_content)
    messages = [prompt]
    
    runs = int(input('how many attempts do you want the LLM to make? '))
    
    tasks = [
        run_single_attempt(messages, i, correct_answer) 
        for i in range(runs)
    ]
    
    results = await asyncio.gather(*tasks)
    
    successes = sum(results)
    
    print(f'The LLM made {runs} attempts and was correct {successes} times')

if __name__ == "__main__":
    asyncio.run(main())
