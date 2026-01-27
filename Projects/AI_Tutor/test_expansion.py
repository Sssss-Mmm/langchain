
from dotenv import load_dotenv
import os

# Set dummy key for testing before importing chains
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing-only"

from state import TutorState
from nodes import generate_question_node
from unittest.mock import MagicMock
import chains

load_dotenv()

# Mocking the chain to avoid API key requirement for logic testing
chains.generate_question_chain = MagicMock()
chains.generate_question_chain.invoke.side_effect = lambda x: f"MOCKED QUESTION (Topic: {x['topic']}, Difficulty: {x['difficulty']}, Type: {x['question_type']})"

def test_generation():
    print("Testing 'conceptual' question generation...")
    state1: TutorState = {
        "topic": "Python Lists",
        "difficulty": "Easy",
        "question_type": "conceptual",
        "messages": [],
        "attempt_count": 0,
        "question_solved": False,
        "last_evaluation": None
    }
    res1 = generate_question_node(state1)
    print(f"Result 1: {res1['current_question']}\n")

    print("Testing 'debugging' question generation...")
    state2: TutorState = {
        "topic": "Python Functions",
        "difficulty": "Medium",
        "question_type": "debugging",
        "messages": [],
        "attempt_count": 0,
        "question_solved": False,
        "last_evaluation": None
    }
    res2 = generate_question_node(state2)
    print(f"Result 2: {res2['current_question']}\n")

if __name__ == "__main__":
    test_generation()
