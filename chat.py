import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import datetime



system_prompt = "Du är en matematiklärare som lär ut matte enbart genom att använda zebror som exempel."
bot_name = "Zebra GPT"
bot_avatar = "🧑‍🏫"


# fetching api key from secrets.toml
openai_api_key = st.secrets["OPENAI_API_KEY"]
# initializing the OpenAI client
client = OpenAI(api_key = openai_api_key)

# fetching api key from secrets.toml
pc_api_key = st.secrets["PINECONE_API_KEY"]
# initializing the Pinecone client
pc = Pinecone(api_key=pc_api_key)

# initializing the messages list
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'index' not in st.session_state:
    st.session_state.index = None

### SIDEBAR START ###

# Sidebar button prompts
st.sidebar.subheader("🔘 **Button Prompts**")

if st.sidebar.button("Send prompt"):
    st.session_state.button_prompt = "Vad är 2+2?"

if st.sidebar.button("Giraffer och zebror"):
    st.session_state.button_prompt = "Hur många ben har en giraff och vem är Elias Musk?"

st.sidebar.markdown("---")

with st.sidebar.expander("Settings", expanded=True):
    # GPT model selection
    st.session_state.gpt_model = st.selectbox(label="⚛️ **GPT Model**", options=["gpt-4-turbo", "gpt-3.5-turbo"], placeholder="Select a model", index=0)

    # Embeddings model selection
    st.session_state.embeddings_model = st.selectbox(label="⚛️ **Embeddings Model**", options=["text-embedding-3-small"], placeholder="Select a model", index=0)

    # Pinecone index selection
    list_indexes = [index.name for index in pc.list_indexes()]
    st.session_state.index = st.selectbox(label="🌲 **Pinecone Index**", options=list_indexes, placeholder="Select an Index", index=None)
    if st.session_state.index is None:
        st.info("💡Select an index to include Pinecone embeddings in the chatbot.")

    # Similarity score threshold
    st.session_state.similarity_score = st.slider("**Similarity Score Threshold**", min_value=0.1, max_value=1.0, value=0.5, step=0.05, help="Lower values might include more context than necessary")


    # clear chat button
    st.sidebar.button("🚮 Clear chat", on_click=lambda: st.session_state.pop("messages", None))
### SIDEBAR END ###

def process_prompt(prompt):
    """Processes the user prompt and generates a response."""
    index_name = st.session_state.index
    if not index_name:
        st.error("No index selected. ")
        return
    
    index = pc.Index(index_name)
    
    initiated_by_button = "button_prompt" in st.session_state and prompt == st.session_state.get("button_prompt")
    st.session_state.messages.append({"role": "user", "content": prompt, "initiated_by_button": initiated_by_button})

    if not initiated_by_button:
        display_message("user", prompt)

    response = None


    with st.chat_message("assistant", avatar=bot_avatar):
        st.write(bot_name)
        with st.spinner(""):
            try:
                res = client.embeddings.create(model=st.session_state.embeddings_model, input=[prompt], encoding_format="float")
                if not res.data:
                    raise ValueError("Failed to create embedding. No data returned.")

                embedding = res.data[0].embedding
                result = index.query(vector=embedding, top_k=5, include_metadata=True)

                if not result.matches:
                    raise ValueError("Ingen vektor matchning hittades.")

                match = result.matches[0]
                if match.score >= st.session_state.similarity_score:
                    context = match.metadata['text']
                    string_before_context = "Använd denna information om det hjälper dig att svara på frågan bättre:"
                else:
                    context = "För tillfället har jag ingen ytterligare information att ge dig, svara utifrån din bästa förmåga."
                    string_before_context = ""

            except Exception as e:
                st.error(f"Error: {e}")
                context = ""
                string_before_context = ""

            system_prompt_full = f"{system_prompt} {string_before_context}{context}"
            print("-" * 40)
            print("New prompt ...")
            print(f"Cosine Similarity threshold: {st.session_state.similarity_score:.2f}")
            print(f"> Vector 1  |  id: {result.matches[0].id}  |  Cos sim: {round(result.matches[0].score, 3)}  |  Context: {' '.join(context.split()[:10])} ...")
            for rank, match in enumerate(result.matches[1:5], start=2):
                print(f"Vector: {rank}  |  id: {match.id}  |  Cos sim: {round(match.score, 3)}")
            print("Fetching response from API...")
            message_list = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            message_list.append({"role": "system", "content": system_prompt_full})
            
            try:
                stream = client.chat.completions.create(
                    model=st.session_state.gpt_model,
                    messages=message_list,
                    temperature=0.1, # (range 0-2) lower values result in more consistent outputs, higher values result in more creative outputs
                    stream=True)
                
            except Exception as e:
                st.error(f"Error in generating response: {e}")
                return
        response = st.write_stream(stream)

    # append the response message to the messages list
    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.toast(f"Similarity Score: {match.score:.3f}")
        if len(st.session_state.messages) > 10:
            st.session_state.messages = st.session_state.messages[-10:]  # keep only the last 10 messages

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # print checks
    print(f"Prompt: ", " ".join(prompt.split()[:10]), " ...")
    print(f"Response: ", " ".join(response.split()[:10]), " ...")
    print(f"Finished. Time: {timestamp}")
    print("-" * 40)




def display_message(role, content):
    """Settings for displaying the messages in the chat window."""
    if role == "assistant":
        avatar = bot_avatar
        display_name = bot_name
    else: 
        avatar = "🙋🏻‍♂️"
        display_name = "**Du:**"

    with st.chat_message(role, avatar=avatar):
        st.write(display_name)
        st.markdown(content)


def main():
    """Main function to run the chatbot."""
    st.header("👨🏻‍🔬 OpenAI - Chatbot")
    st.subheader("🔍📑 Load your Pinecone index in the sidebar", divider="green")
    st.markdown("\n \n")


    # displays the messages in the chat window
    messages_to_display = [message for message in st.session_state.messages if not message.get("initiated_by_button", False)]
    for message in messages_to_display:
        display_message(message["role"], message["content"])

    # walrus operator to check if the user has entered a prompt and then processes it
    if prompt := st.chat_input("Skriv din fråga här"):
        process_prompt(prompt)

    # check if a button prompt is in the session state and process it
    if 'button_prompt' in st.session_state:
        prompt_to_process = st.session_state.button_prompt
        process_prompt(prompt_to_process)
        del st.session_state.button_prompt  # clear after processing


if __name__ == "__main__":
    main()