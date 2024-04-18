import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import time
import datetime

# setting the page config
st.set_page_config(
    page_title="Home", 
    page_icon="🤖",
    layout="centered"
    )

# init session states
if 'openai' not in st.session_state:
    st.session_state.openai = None
if 'pinecone' not in st.session_state:
    st.session_state.pinecone = None
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'gpt_model' not in st.session_state:
    st.session_state.gpt_model = None


st.session_state.openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.session_state.pinecone = Pinecone(api_key=st.secrets["PC_API_KEY"])

st.session_state.embeddings_model = "text-embedding-3-small"
st.session_state.gpt_model = "gpt-4-0125-preview"


def intro():
    st.header('Welcome to BK24TR Hackathon! 🚀', divider="green")
    st.markdown("\n \n")

    # Description
    st.write("""
    This app demonstrates the use of Streamlit, OpenAI API, and Pinecone Serverless Vector database 
    to test RAG (Retrieval-Augmented Generation) functionality. With this app, you can interact with 
    an OpenAI GPT-4 model via chatbot interface and utilize Pinecone for managing indexes and upserting records.

    To get started, you'll need API keys from OpenAI and Pinecone.
    """)

    # Instructions
    st.header('Instructions:')
    st.write("""
    1. Obtain API keys from OpenAI and Pinecone.
    2. Enter the keys in the appropriate fields.
    4. Create a new index in Pinecone.
    5. Upsert vectors to the index.
    6. Start testing RAG functionality by chatting with the OpenAI GPT-4 model.
    """)

    "---"

    # 3 columns with logos from streamlit, openai and pinecone
    spacer1, col1, col2, col3 = st.columns([1, 2, 2, 2])
    with col1:
        st.markdown("\n \n")
        st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=125)
    with col2:
        st.image("https://openai.com/favicon.ico", width=100)
    with col3:
        st.image("https://pbs.twimg.com/profile_images/1676450951874453505/y5_T5OWH_200x200.png", width=100)


    # Sidebar
    st.sidebar.subheader("👆🏼 Select a page from the dropdown menu to get started.")

    st.sidebar.markdown("---")

    # Input fields for API keys
    openai_key = st.sidebar.text_input('Enter your OpenAI API Key:', placeholder="Temporarily disabled", disabled=True)
    pinecone_key = st.sidebar.text_input('Enter your Pinecone API Key:', placeholder="Temporarily disabled", disabled=True)



def list_indexes():
    """List all indexes in Pinecone. Used in multiple pages."""
    pc = st.session_state.pinecone
    with st.sidebar.expander("↗ View Indexes", expanded=True):
        list_indexes = pc.list_indexes()
        total_indexes = len(list_indexes)
        for i, index in enumerate(list_indexes):
            st.subheader(f"📇 **:green[{index.name}]**")
            with st.container():
                st.write(f"**Dimensions:** {index.dimension}")
                st.write(f"**Metric:** {index.metric}")
                status_ready = index.status.get('ready', False)
                status = "Ready" if status_ready else "Not Ready"
                st.write(f"**Status:** {status}")
                if i < total_indexes - 1:
                    st.markdown("\n \n")

def pc_create_index():
    """Create a new index in Pinecone."""
    pc = st.session_state.pinecone

    embeddings_model = "text-embedding-3-small"

    st.header("🌲 Pinecone - Create Index", divider="green")
    st.markdown("\n \n")


    with st.form(key="create_index_form", clear_on_submit=True):
        new_index_name = st.text_input("Index name")
        dimension = st.selectbox("Dimension", [1536], help=f"For {embeddings_model}: **1536** is recommended")
        metric = st.selectbox("Metric", ["cosine", "euclidean"], help=f"For {embeddings_model}: **Cosine** is recommended")
        cloud = st.selectbox("Cloud", ["aws", "gcp"])
        region = st.selectbox("Region", ["us-east-1", "us-west-2", "eu-central-1"])
        st.caption("The default settings are recommended for this use case")
        index_create_submit_button = st.form_submit_button(label="Create Index")
        # Check if the submit button is pressed
        try:
            if index_create_submit_button:
                # Create the index
                pc.create_index(
                    name=new_index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )
                # Display a success message
                st.success(f"Index {new_index_name} created successfully!")
        except Exception as e:
            if "already exists" in str(e):
                st.error(f"**Error 409:** Index :blue[{new_index_name}] already exists.")

    # lists the available indexes in the sidebar
    list_indexes()

def pc_upsert():
    """Upsert vectors to an existing index in Pinecone."""
    pc = st.session_state.pinecone
    client = st.session_state.openai
    embeddings_model = st.session_state.embeddings_model

    st.header("🌲 Pinecone - Upsert Vectors", divider="green")
    st.markdown("\n \n")


    if 'embeds' not in st.session_state:
        st.session_state.embeds = None

    with st.form(key="upsert_vectors_form", clear_on_submit=True):
        indexes = pc.list_indexes()
        index_name = st.selectbox(label="Index name", options=[index.name for index in indexes], placeholder="Select an index", index=None)
        vector = st.text_area(label="Vector", placeholder="This is some very specific information our LLM is unaware of..", help="Enter the text to generate the embedding")
        vector_id = st.text_input(label="Vector ID", help="Must be unique")
        embed_vector_submit_button = st.form_submit_button(label="Generate Embedding")
    
    # if vector_id already exists in index print already exists'
    if 'id_exists' not in st.session_state:
        st.session_state.id_exists = True

    if embed_vector_submit_button:
        index = pc.Index(index_name)
        try:
            result = index.query(
                namespace="",
                id=vector_id,
                top_k=1,
                include_metadata=True
            )
            if result.matches[0].id == vector_id:
                st.error(f"**Error 409:** Vector ID :blue[{vector_id}] already exists.")
        except Exception as e:
            st.session_state.id_exists = False
            

    if embed_vector_submit_button and vector and vector_id and not st.session_state.id_exists:
        try:
            # Create the embedding
            res = client.embeddings.create(
                model=embeddings_model, 
                input=[vector], 
                encoding_format="float")
            st.session_state.embeds = [record.embedding for record in res.data]
            sample_embeds = f"{st.session_state.embeds[0][:5]} ... {st.session_state.embeds[0][-5:]}"
            st.write(f"Embedding: {sample_embeds}")
        except Exception as e:
            st.error(f"**Error:** {e}")

    
    # Confirm button to upsert the embedding
    if st.session_state.embeds:
        if st.button("Upsert to Index"):
            index = pc.Index(index_name)
            try:
                index.upsert(
                    vectors=[(vector_id, st.session_state.embeds[0], {"text": vector})],
                )
                st.toast("Vector upserted successfully!")
            except Exception as e:
                st.error(f"**Error:** {e}")

    # lists the available indexes in the sidebar
    list_indexes()

def pc_query():
    """Query an existing index in Pinecone."""
    pc = st.session_state.pinecone

    st.header("🌲 Pinecone - Query Index", divider="green")
    st.markdown("\n \n")

    indexes = pc.list_indexes()
    index_names = [index.name for index in indexes] 
 
    with st.form(key="query_index_form", clear_on_submit=True):
        index_name = st.selectbox("Index name", index_names, placeholder="Select an index", index=None)
        top_k_value = st.slider("Top K", 1, 5)
        query_index_submit_button = st.form_submit_button("Query")
        if query_index_submit_button:
            index = pc.Index(index_name)
            try:
                # Query the index
                result = index.query(
                    namespace="",
                    id="1",
                    top_k=top_k_value,
                    include_metadata=True
                )
                st.write(f"Result: {result}")
            except Exception as e:
                st.error(f"**Error:** {e}")

    
    # lists the available indexes in the sidebar
    list_indexes()


def openai_chatbot():
    """OpenAI Chatbot using Pinecone for embeddings."""
    # model settings
    gpt_model = "gpt-4-0125-preview"
    embeddings_model = "text-embedding-3-small"
    system_prompt = "Hej! Jag är en matematikassistent som kan hjälpa dig med dina frågor. Vad vill du veta?"
    bot_name = "Matematik GPT"


    # fetching api key from secrets.toml
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    # initializing the OpenAI client
    client = OpenAI(api_key = openai_api_key)

    # fetching api key from secrets.toml
    pc_api_key = st.secrets["PC_API_KEY"]
    # initializing the Pinecone client
    pc = Pinecone(api_key=pc_api_key)

    # initializing the messages list
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'index' not in st.session_state:
        st.session_state.index = None
    
    list_indexes = [index.name for index in pc.list_indexes()]
    st.session_state.index = st.sidebar.radio("Select an index", list_indexes)
    


    def process_prompt(prompt):
        """Processes the user prompt and generates a response."""
        index_name = st.session_state.index
        if not index_name:
            st.error("No index selected.")
            return
        
        index = pc.Index(index_name)
        
        initiated_by_button = "button_prompt" in st.session_state and prompt == st.session_state.get("button_prompt")
        st.session_state.messages.append({"role": "user", "content": prompt, "initiated_by_button": initiated_by_button})

        if not initiated_by_button:
            display_message("user", prompt)

        response = None


        with st.chat_message("assistant", avatar="🧑‍🏫"):
            st.write("**Matematik GPT:**")
            with st.spinner(""):
                try:
                    res = client.embeddings.create(model=embeddings_model, input=[prompt], encoding_format="float")
                    if not res.data:
                        raise ValueError("Failed to create embedding. No data returned.")

                    embedding = res.data[0].embedding
                    result = index.query(vector=embedding, top_k=5, include_metadata=True)

                    if not result.matches:
                        raise ValueError("Ingen vektor matchning hittades.")

                    match = result.matches[0]
                    cosine_threshold = 0.1
                    if match.score >= cosine_threshold:
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
                print(f"Cosine Similarity threshold: {cosine_threshold}")
                print(f"> Vector 1  |  id: {result.matches[0].id}  |  Cos sim: {round(result.matches[0].score, 3)}  |  Context: {' '.join(context.split()[:10])} ...")
                for rank, match in enumerate(result.matches[1:5], start=2):
                    print(f"Vector: {rank}  |  id: {match.id}  |  Cos sim: {round(match.score, 3)}")
                print("Fetching response from API...")
                message_list = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                message_list.append({"role": "system", "content": system_prompt_full})
                
                try:
                    stream = client.chat.completions.create(
                        model=gpt_model,
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
            avatar = "🧑‍🏫"
            display_name = bot_name
        else: 
            avatar = "🙋🏻‍♂️"
            display_name = "**Du:**"

        with st.chat_message(role, avatar=avatar):
            st.write(display_name)
            st.markdown(content)


    def main():
        """Main function to run the chatbot."""
        st.title("Matematik GPT 🤖")

        # all the sidebar funcionality inside this with block
        with st.sidebar:

            # clear chat button
            st.button("Rensa chatten", on_click=lambda: st.session_state.pop("messages", None))


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


page_names_to_funcs = {
"🏠 Home": intro,
"🌲 Pinecone - Create Index": pc_create_index,
"🌲 Pinecone - Upsert Vectors": pc_upsert,
"🌲 Pinecone - Query Index": pc_query,
"🧠 OpenAI - Chatbot": openai_chatbot
}

demo_name = st.sidebar.selectbox("Page Selection:", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()