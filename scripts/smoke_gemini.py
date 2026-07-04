"""Phase-1 gate: prove Gemini function calling works end to end with one tool, one call.

Run: python scripts/smoke_gemini.py

This verifies the live SDK shape (how function calls and usage metadata come back)
before any Witness code is built on top of it. It prints the exact response structure
so the LLM client can be written against reality, not stale memory of the SDK.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        return 2

    import google.generativeai as genai

    print(f"google-generativeai version: {genai.__version__}")
    genai.configure(api_key=api_key)

    model_name = "gemini-2.5-flash-lite"

    # One tool declaration in the JSON-schema shape Gemini expects for function calling.
    tools = [
        {
            "function_declarations": [
                {
                    "name": "get_ticket_status",
                    "description": "Look up the status of a support ticket by its numeric id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticket_id": {
                                "type": "integer",
                                "description": "The numeric ticket id to look up.",
                            }
                        },
                        "required": ["ticket_id"],
                    },
                }
            ]
        }
    ]

    model = genai.GenerativeModel(
        model_name=model_name,
        tools=tools,
        system_instruction="You are a support assistant. Use tools when a lookup is needed.",
    )

    prompt = "What is the status of ticket 4471? Use the tool."
    print(f"\nMODEL: {model_name}\nPROMPT: {prompt}\n")

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.0),
    )

    # Inspect the response structure so we know exactly how to parse it in core/llm.py.
    candidate = response.candidates[0]
    print(f"finish_reason: {candidate.finish_reason}")
    print(f"parts count: {len(candidate.content.parts)}")

    found_function_call = False
    for i, part in enumerate(candidate.content.parts):
        fc = getattr(part, "function_call", None)
        text = getattr(part, "text", None)
        if fc and fc.name:
            found_function_call = True
            # fc.args behaves like a dict (proto MapComposite); coerce to a plain dict.
            args = {k: v for k, v in fc.args.items()}
            print(f"  part[{i}] FUNCTION_CALL name={fc.name!r} args={args!r}")
        elif text:
            print(f"  part[{i}] TEXT {text!r}")
        else:
            print(f"  part[{i}] (other) {part!r}")

    um = response.usage_metadata
    print(
        "\nusage_metadata: "
        f"prompt={um.prompt_token_count} "
        f"candidates={um.candidates_token_count} "
        f"total={um.total_token_count}"
    )

    if found_function_call:
        print("\nSMOKE TEST PASSED: function calling works and the shape is as expected.")
        return 0

    print("\nSMOKE TEST WARNING: no function_call in response; model answered directly.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
