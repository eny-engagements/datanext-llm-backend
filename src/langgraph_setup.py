import os
import shutil

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.supervisor import master_agent_graph, members
from src.config.settings import settings

os.makedirs(settings.DEFINE_AI_OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.LINEAGE_AI_OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.DISCOVER_AI_OUTPUT_DIR, exist_ok=True)


# Step 4
# Helper for creating dummy files for testing
def create_dummy_file(
    filename, content="dummy content", directory="supervisor_test_inputs"
):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


if __name__ == "__main__":
    # Clean up for fresh run
    if os.path.exists("supervisor_test_inputs"):
        shutil.rmtree("supervisor_test_inputs")
    if os.path.exists(settings.DEFINE_AI_OUTPUT_DIR):
        shutil.rmtree(settings.DEFINE_AI_OUTPUT_DIR)
    if os.path.exists(settings.LINEAGE_AI_OUTPUT_DIR):
        shutil.rmtree(settings.LINEAGE_AI_OUTPUT_DIR)
    if os.path.exists(settings.DISCOVER_AI_OUTPUT_DIR):
        shutil.rmtree(settings.DISCOVER_AI_OUTPUT_DIR)
    os.makedirs("supervisor_test_inputs", exist_ok=True)
    os.makedirs(settings.DEFINE_AI_OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.LINEAGE_AI_OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.DISCOVER_AI_OUTPUT_DIR, exist_ok=True)

    # Conversational Loop
    conversation_history = []

    while True:
        user_input = input(">>> User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation.")
            break

        conversation_history.append(HumanMessage(content=user_input))

        # If the user mentions uploading a file, we simulate it here for the tools to find.
        # In a real UI, the UI would handle the upload and provide the path.
        # For this demo, if user says "using schema.xlsx", we create it.
        # This is a simplification; robust file path handling from text is hard.
        # The supervisor prompt instructs it to ASK for paths if unclear.
        if "schema.xlsx" in user_input.lower():
            create_dummy_file(
                "schema.xlsx",
                "Schema Name,Table Name,Column Name\nSALES,ORDERS,ORDER_ID",
                directory="supervisor_test_inputs",
            )
            print(
                "(Simulated upload of schema.xlsx to supervisor_test_inputs/schema.xlsx)"
            )
        if "etl_script.sql" in user_input.lower():
            create_dummy_file(
                "etl_script.sql",
                "SELECT * FROM users;",
                directory="supervisor_test_inputs",
            )
            print(
                "(Simulated upload of etl_script.sql to supervisor_test_inputs/etl_script.sql)"
            )
        if "glossary.xlsx" in user_input.lower():
            create_dummy_file(
                "glossary.xlsx",
                "Table Name,Column Name,Description\nusers,user_id,User ID",
                directory="supervisor_test_inputs",
            )
            print(
                "(Simulated upload of glossary.xlsx to supervisor_test_inputs/glossary.xlsx)"
            )

        print("\n--- Master Agent Processing ---")
        # The supervisor expects a list of messages
        # For the stream, the input should be a dictionary with a "messages" key
        events = master_agent_graph.stream({"messages": conversation_history})
        ai_response_content = ""  # To store the last user-facing AI message
        final_event_message = None
        print("\n--- Raw Stream Events ---")
        for event_dict in events:
            print(event_dict)

            for agent_name, agent_output in event_dict.items():
                if (
                    isinstance(agent_output, dict)
                    and "messages" in agent_output
                    and agent_output["messages"]
                ):
                    final_event_messages = agent_output["messages"]
        print("--- End Raw Stream Events ---\n")

        if final_event_messages:
            last_msg_in_event = final_event_messages[-1]
            if isinstance(last_msg_in_event, AIMessage):

                if (
                    last_msg_in_event.content not in members
                    and last_msg_in_event.content != "supervisor"
                ):
                    ai_response_content = last_msg_in_event.content

        if ai_response_content:
            print(f"<<< AI: {ai_response_content}")
            conversation_history.append(
                AIMessage(
                    content=ai_response_content,
                    name=last_msg_in_event.name or "supervisor",
                )
            )
        else:
            print(
                "<<< AI: (No explicit user-facing message to display from this turn's events. The supervisor might be routing or an agent is processing.)"
            )
            # If no user-facing message, but supervisor  produced some AIMessage (like a routing instruction),

        print("--- Turn End ---\n")
