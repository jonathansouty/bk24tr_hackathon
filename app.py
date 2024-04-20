import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import time
import datetime

# setting the page config
st.set_page_config(
    page_title="RAG Builder", 
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
st.session_state.pinecone = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

st.session_state.embeddings_model = "text-embedding-3-small"
st.session_state.gpt_model = "gpt-4-0125-preview"


def intro():
    st.header('🌎 RAG Builder App', divider="green")
    st.markdown("\n \n")

    st.write("""
    This app demonstrates the use of Streamlit, OpenAI API, LangChain and Pinecone Serverless Vector database 
    to build RAG (Retrieval-Augmented Generation). With this app, you can interact with 
    an OpenAI GPT-4 model via chatbot interface and utilize Pinecone for managing indexes and upserting records.

    To get started, you'll need API keys from OpenAI and Pinecone.
    """)

    st.subheader("👈 Go to the instructions page to get started.")

    st.subheader("🔗 Useful Links:")
    st.page_link("https://chunkviz.up.railway.app/", label="👉 ChunkViz")
    st.write("ChunkViz is a tool to help you visualize and understand chunking embeddings.")
    st.caption("Credit to Greg Kamradt for open-sourcing ChunkViz.")

    "---"

    st.write("Feel free to fork the repository on [GitHub]")
    st.caption("This app is made by Jonathan Souty")
    
    st.markdown("\n \n")


    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        st.markdown("\n \n")
        st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=125)
    with col2:
        st.image("https://openai.com/favicon.ico", width=100)
    with col3:
        st.image("https://pbs.twimg.com/profile_images/1676450951874453505/y5_T5OWH_200x200.png", width=100)
    with col4:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/LangChain_logo.svg/640px-LangChain_logo.svg.png", width=180)


    # Sidebar
    st.sidebar.markdown("---")

    # Input fields for API keys
    openai_key = st.sidebar.text_input('Enter your OpenAI API Key:', placeholder="Temporarily disabled", disabled=True)
    pinecone_key = st.sidebar.text_input('Enter your Pinecone API Key:', placeholder="Temporarily disabled", disabled=True)

def rag_info():
    """Information about the app and instructions for getting started"""
    st.sidebar.markdown("---")
    st.header("📚 Getting started with RAG", divider="green")
    st.markdown("\n \n")

    st.subheader("Pinecone Workflow:")
    st.markdown("**Getting started with Pincone:**")
    st.write(" Create a new vector database by specifying the dimension of the vectors you'll be storing and the metric (eg. cosine, euclidean distance, dot product). You can then start uploading your vectors using the upsert method, where each vector is associated with a unique ID. Finally, you can query the database by sending vector queries to retrieve the most similar vectors based on your specified metrics. Pinecone also supports more advanced features like metadata filtering and batch querying, enabling powerful and efficient data retrieval in your applications.")
    st.image("https://i.postimg.cc/8CjCcJTM/vectordb.png", use_column_width=True)

    "---"

    st.subheader("RAG Workflow:")
    st.write("In this specific example, the user asks about servicing a model of forklift, and the RAG system provides a detailed list of parts needed for servicing the brakes on that forklift model. The RAG system retrieves the relevant information from a knowledge base and generates a response based on the user's query. The system uses a combination of retrieval and generation techniques to provide accurate and informative responses to user queries.")
    st.image("https://i.postimg.cc/rFfc4VnJ/ragmindmap.png", use_column_width=True)
    rag_explanation = """
    **1. User Prompt:** The process begins with a user asking a question, such as inquiring about the specific servicing requirements for a model of forklifts.

    **2. Embedding Model:** This user prompt is processed through an embedding model which converts the text into a numerical vector that represents the semantic content of the prompt.

    **3. Numerical Vectors:** The output from the embedding model is a set of numerical vectors, which encode the meaning of the user prompt into a form that can be compared and searched against other vectors.

    **4. Vector Database (Pinecone):** These vectors are then used to perform a query in a vector database (powered by Pinecone). The database contains a large collection of pre-indexed vectors from various sources.

    **5. Similarity Search:** The database performs a similarity search to find vectors that are closest to the user prompt vector, essentially finding the most relevant information related to the query.

    **6. User Prompt + Context:** The relevant vectors found are used to construct context information which is appended to the original user prompt. This enriches the prompt with details relevant to the user's query.

    **7. GPT Model:** The enriched prompt is then fed into a GPT model, which uses both the original prompt and the additional context to generate a detailed, accurate response.

    **8. Response:** Finally, the GPT model outputs a response that is informed by the context retrieved from the vector database, answering the user's question in a knowledgeable way.
    """
    with st.expander("📖 RAG Explanation - Click to expand", expanded=False):
        st.markdown(rag_explanation)

    "---"

    st.subheader("🔗🦜 LangChain Loaders:")


def list_all_indexes():
    """List all indexes in Pinecone. Used in multiple pages."""
    pc = st.session_state.pinecone
    with st.sidebar.expander("↗ View Indexes", expanded=False):
        list_indexes_view = pc.list_indexes()
        index_name_view = st.selectbox("Indexes", [index.name for index in list_indexes_view], placeholder= "Select an index", index=None)
        if index_name_view is not None:
            index = pc.describe_index(index_name_view)
            index_describe = pc.Index(index_name_view)
            describe = index_describe.describe_index_stats()
            st.write(f"**Index:** {index_name_view}")
            st.write(f"**Dimensions:** {index.dimension}")
            st.write(f"**Metric:** {index.metric}")
            st.write(f"**Vectors:** {describe['total_vector_count']}")

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
        st.caption(f"ⓘ The default settings are recommended for the selected embeddings model: {embeddings_model}")
        index_create_submit_button = st.form_submit_button(label="Create Index")
        # Check if the submit button is pressed
        try:
            if index_create_submit_button:
                with st.spinner("Creating index... ⌛"):
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
                    time.sleep(2)
                    st.success(f"{new_index_name} was created successfully!")
                    time.sleep(1.5)
                    st.rerun()

        except Exception as e:
            if "already exists" in str(e):
                st.error(f"**Error 409:** Index :blue[{new_index_name}] already exists.")

    st.sidebar.markdown("---")

    # lists the available indexes in the sidebar
    list_all_indexes()

    with st.sidebar:
        with st.expander("❌ Delete Index", expanded=False):
            if 'index_to_delete' not in st.session_state:
                st.session_state.index_to_delete = None

            index_name_delete = st.selectbox("Index name", [index.name for index in pc.list_indexes()], placeholder="Select index to delete", index=None)
            if index_name_delete is not None:
                st.session_state.index_to_delete = index_name_delete
                st.warning("Bad things can happen if you delete an index. Deleting an index is irreversible, be mindful of your selection.")

            if st.session_state.index_to_delete:
                with st.form(key="delete_index_form", clear_on_submit=True, border=False):
                    delete_index_input = st.text_input(f"Type '{index_name_delete}' to confirm deletion")
                    delete_index_submit_button = st.form_submit_button("Delete Index")
                    
                    if delete_index_submit_button:
                        if delete_index_input == st.session_state.index_to_delete:
                            with st.spinner(f"Deleting index ... 🗑️"):
                                pc.delete_index(st.session_state.index_to_delete)
                                time.sleep(1.5)
                                st.success(f"Deleted index: {st.session_state.index_to_delete}")
                                del st.session_state.index_to_delete
                                time.sleep(1)
                                st.rerun()
                        else:
                            st.error("Wrong input. Please type the index name to confirm deletion.")



def pc_upsert():
    """Upsert vectors to an existing index in Pinecone."""
    pc = st.session_state.pinecone
    client = st.session_state.openai
    embeddings_model = st.session_state.embeddings_model

    st.header("🌲 Pinecone - Upsert Vectors")

    if 'embeds' not in st.session_state:
        st.session_state.embeds = None

    with st.form(key="upsert_vectors_form", clear_on_submit=True, border=False):
        indexes = pc.list_indexes()
        if not indexes:
            st.error("No indexes available.")
            return
        index_options = [index.name for index in indexes]
        index_name_upsert = st.selectbox("Index name", options=index_options, placeholder="Select an index", index=None)
        vector = st.text_area("Vector", help="Enter the text to generate the embedding")
        vector_id = st.text_input("Vector ID", help="Must be unique")
        embed_vector_submit_button = st.form_submit_button("🆙 Upsert embeddings to index")
    
    if embed_vector_submit_button:
        if not vector or not vector_id:
            st.error("Vector and Vector ID must be provided.")
            return

        index = pc.Index(index_name_upsert)
        existing_ids = [match.id for match in index.query(id=vector_id, top_k=1).matches]
        if vector_id in existing_ids:
            st.error(f"**Error 409:** Vector ID {vector_id} already exists.")
            return

        try:
            res = client.embeddings.create(model=embeddings_model, input=[vector], encoding_format="float")
            st.session_state.embeds = [record.embedding for record in res.data]
            sample_embeds = f"{st.session_state.embeds[0][:5]} ... {st.session_state.embeds[0][-5:]}"
            st.write(f"Vector text: {vector[:50]}...")
            st.write(f"Embedding sample: {sample_embeds}")
        except Exception as e:
            st.error(f"Failed to create embeddings: {e}")
            return
        
        try:
            index.upsert(vectors=[(vector_id, st.session_state.embeds[0], {"text": vector})])
            st.success("Vector upserted successfully!")
            if st.button("🔄 Clear"):
                del st.session_state.embeds
                time.sleep(1.5)
                st.rerun()
        except Exception as e:
            st.error(f"Failed to upsert vector: {e}")

    st.sidebar.markdown("---")

    # lists the available indexes in the sidebar
    list_all_indexes()

    with st.sidebar:

        with st.expander("🚮 Delete all vectors from an index", expanded=False):
            indexes = pc.list_indexes()  # Retrieve the list of Pinecone indexes
            index_names = [index.name for index in indexes]
            
            # Create a select box for choosing an index from which to delete all vectors
            index_name_to_delete_vectors = st.selectbox("Select an index to delete vectors from", index_names, placeholder="Select an index", index=None)

            # Create a form for deletion confirmation
            with st.form(key="delete_vectors_form", clear_on_submit=True, border=False):
                confirmation_text = st.text_input("Type 'delete all vectors' to confirm deletion")

                # Delete button
                delete_button = st.form_submit_button("Delete Vectors")

                if delete_button and confirmation_text == "delete all vectors":
                    if index_name_to_delete_vectors:
                        index = pc.Index(index_name_to_delete_vectors)  # Initialize the index from which to delete vectors
                        vector_count = index.describe_index_stats()['total_vector_count']
                        ids_to_delete = [ids for ids in index.list(namespace="")]  # List all vector IDs in the index
                        for ids in ids_to_delete:
                            index.delete(ids)
                        st.success(f"Deleted {vector_count} vectors from index: {index_name_to_delete_vectors}")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error("Please select an index to delete vectors from.")
                elif delete_button:
                    st.error("Please type the exact confirmation text to proceed with deletion.")


def pc_query():
    """Query an existing index in Pinecone."""

    st.session_state.openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    st.session_state.pinecone = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

    pc = st.session_state.pinecone
    client = st.session_state.openai

    st.header("🌲 Pinecone - Query Index", divider="green")
    st.markdown("\n \n")

    indexes = pc.list_indexes()
    index_names = [index.name for index in indexes] 
 
    with st.form(key="query_index_form", clear_on_submit=True, border=False):
        index_to_query = st.selectbox("Index name", index_names, placeholder="Select an index", index=None)
        expander_index = index_to_query
        query_text = st.text_area("Query", help="Enter the text to query the index")
        top_k_value = st.slider("Top K", 1, 5)
        query_index_submit_button = st.form_submit_button("Query")
        if query_index_submit_button and index_to_query and query_text:
            index_to_query = pc.Index(index_to_query)
            res = client.embeddings.create(
                model=st.session_state.embeddings_model,
                input=[query_text], 
                encoding_format="float"
                )
            embedding = res.data[0].embedding
            result = index_to_query.query(
                vector=embedding,
                top_k=top_k_value,
                include_metadata=True
            )
            st.markdown("\n \n")
            with st.expander(f"🔍 Query Results for {expander_index}", expanded=True):
                st.write(f"**Top {top_k_value} matches:**")
                st.markdown("\n \n")
                for i, match in enumerate(result.matches[:top_k_value], start=1):
                    st.write(f"**Vector Match: {i}**")
                    st.write(f"ID: {match.id}")
                    st.write(f"Score: :blue[{round(match.score, 3)}]")
                    st.write(f"Metadata: {match.metadata}")
                    if i < top_k_value:
                        st.markdown("---")

    if query_index_submit_button:    
        if st.button("🔄 Clear"):
            st.rerun()

    st.sidebar.markdown("---")        
    # lists the available indexes in the sidebar
    list_all_indexes()


def lc_loaders():
    """Upsert vectors to an existing index using LangChain."""
    from langchain_pinecone import PineconeVectorStore
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
    import tempfile
    

    embeddings = OpenAIEmbeddings()
    #pc = st.session_state.pinecone
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

    st.header("🦜🔗 LangChain - Document Loaders", divider="green")
    st.markdown("\n \n")
    st.caption("💡 Tip: Use [ChunkViz](https://chunkviz.up.railway.app/) to visualize how your text will be split into chunks.")
    with st.expander("📢 Difference between splitters", expanded=False):
        st.caption("It's important to understand the difference between the two chosen splitters.")
        st.write("**CharacterTextSplitter:** Splits the text into chunks based on a specified separator (if chosen, otherwise purely on word count).")
        st.write("**RecursiveCharacterTextSplitter:** Splits the text into chunks recursively, ensuring that each chunk is a complete sentence.")

    indexes = pc.list_indexes()
    index_name = st.selectbox(label="Index name", options=[index.name for index in indexes], placeholder="Select an index", index=None)

    col1, col2 = st.columns([5, 1])
    with col1:
        splitter_type = st.selectbox("Text Splitter", ["CharacterTextSplitter", "RecursiveCharacterTextSplitter"], index=0)

    if splitter_type == "RecursiveCharacterTextSplitter":
        with col2:
            st.text_input("Separator", value="Disabled", disabled=True)
    
    separator = ""
    if splitter_type == "CharacterTextSplitter":
        with col2:
            separator = st.text_input("Separator", help="Choose a separator to split the text. Can be anything you want including empty.")
            if separator == "":
                separator = None  # Handle case where no separator is provided

    # chunk size
    chunk_size = st.slider("Chunk Size", 128, 1024, 512)

    # chunk overlap
    chunk_overlap = st.slider("Chunk Overlap", 0, 256, 0)

    # file uploader
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name


    if st.button("Upsert Vectors", disabled=True):
        # Initialize progress bar
        progress_bar = st.progress(0)

        # Load the documents
        loader = TextLoader(temp_file_path, encoding="utf-8")
        documents = list(loader.load())  # Ensure documents are loaded into memory


        # Initialize the appropriate text splitter
        if splitter_type == "CharacterTextSplitter":
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator, is_separator_regex=False)
        elif splitter_type == "RecursiveCharacterTextSplitter":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)

        total_docs = len(docs)
        for i, doc_chunk in enumerate(docs):
            vectorstore_from_docs = PineconeVectorStore.from_documents(
                [doc_chunk],
                index_name=index_name,
                embedding=embeddings
            )

            # Update progress bar
            progress_percentage = (i + 1) / total_docs
            progress_bar.progress(progress_percentage, f"⌛ Upserting vectors... {progress_percentage:.0%}, {i + 1}/{total_docs} documents processed.")

        # Complete the progress bar when done
        progress_bar.empty()
        st.success(f"Upserted {total_docs} vectors to {index_name}")
    
    st.sidebar.markdown("---")
    # lists the available indexes in the sidebar
    list_all_indexes()


def lc_query():
    """Query an existing index using LangChain."""
    from langchain_pinecone import PineconeVectorStore
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
    import tempfile

    # under construction
    st.header("🦜🔗 LangChain - Query Index", divider="green")
    st.markdown("\n \n")

    st.subheader("🚧👷🏗️ Under construction")


def openai_chatbot():
    """OpenAI Chatbot using Pinecone for embeddings."""
    # model settings

    system_prompt = "Hej! Jag är en matematikassistent som kan hjälpa dig med dina frågor. Vad vill du veta?"
    bot_name = "Matematik GPT"
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
    st.sidebar.markdown("---")
    st.sidebar.header("Settings")

    # GPT model selection
    st.session_state.gpt_model = st.sidebar.selectbox(label="⚛️ **GPT Model**", options=["gpt-4-turbo", "gpt-3.5-turbo"], placeholder="Select a model", index=0)

    # Embeddings model selection
    st.session_state.embeddings_model = st.sidebar.selectbox(label="⚛️ **Embeddings Model**", options=["text-embedding-3-small"], placeholder="Select a model", index=0)

    # Pinecone index selection
    list_indexes = [index.name for index in pc.list_indexes()]
    st.session_state.index = st.sidebar.selectbox(label="🌲 **Pinecone Index**", options=list_indexes, placeholder="Select an Index", index=None)
    if st.session_state.index is None:
        st.sidebar.info("💡Select an index to include Pinecone embeddings in the chatbot.")
    
    st.sidebar.markdown("---")

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


page_names_to_funcs = {
"🏠 Home": intro,
"📚 Getting started with RAG": rag_info,
"🌲 Pinecone - Create Index": pc_create_index,
"🌲 Pinecone - Upsert Vectors": pc_upsert,
"🌲 Pinecone - Query Index": pc_query,
"🦜🔗 LangChain - Document Loaders": lc_loaders,
"🦜🔗 LangChain - Query Index": lc_query,
"⚛️ OpenAI - Chatbot": openai_chatbot
}

demo_name = st.sidebar.radio("Page Selection:", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()