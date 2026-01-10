from openai import OpenAI
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from pydantic import BaseModel


class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


# Use mlx-batch-server to provide local OpenAI service
client = OpenAI(
    base_url="http://localhost:8100/v1",
    api_key="mlx-batch-server",  # not-needed
)

structured_output_agent = Agent(
    model=OpenAIChat(
        client=client,
        id="mlx-community/SmallThinker-3B-Preview-4bit",
    ),
    description="You are a helpful math tutor. Guide the user through the solution step by step.",
    response_model=MathReasoning,
    structured_outputs=True,
)

# Run the agent synchronously
structured_output_agent.print_response("how can I solve 8x + 7 = -23")
