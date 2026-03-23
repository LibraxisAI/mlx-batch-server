from pydantic import BaseModel
from rich import print as rprint


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


from openai import OpenAI

# Use mlx-batch-server to provide local OpenAI service
client = OpenAI(
    base_url="http://localhost:10240/v1",
    api_key="mlx-batch-server",  # not-needed
)

response = client.beta.chat.completions.parse(
    model="mlx-community/SmallThinker-3B-Preview-4bit",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
        },
        {"role": "user", "content": "how can I solve 8x + 7 = -23"},
    ],
    response_format=MathReasoning,
)

# Get generated content
result = response.choices[0].message.parsed
rprint(result)
