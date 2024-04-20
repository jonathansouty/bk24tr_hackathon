# BK24TR Hackathon

## Description
This project is a hackathon for class BK24TR.<br>
The goal is to optimize an AI assistant for teaching math to 4th graders using a Streamlit webapp that utilizes OpenAI API and Pinecone.
<br>

## Prerequisites
Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).<br>
Aswell as an IDE, we will be using Visual Studio Code as an example in the instructions.
<br>

## Setup
1. Fork the repository
2. Follow the instructions
The following is a standard guide on how to do it on Windows, using Anaconda instead is completely fine aswell.
<br>

## Installation

### Create a new virtual environment
When running this line in the terminal, make sure your current directory is where you want the venv to be located.
Avoid creating it inside a github repo folder.
```bash
# Create a virtual environment (outside your project directory)
python -m venv bk24tr_env

# For Windows:
bk24tr_env\Scripts\activate
```

### Cloning the Repository
Clone the repository using Git and choose which current directory you want as the repository folder.
Be mindful of where your 'cd' (current directory) path is set to before running the cd bk24tr_hackathon line.
```bash
git clone https://github.com/jesab/bk24tr_hackathon
cd bk24tr_hackathon
```
Its also possible to do this in VS Code if you prefer a more visual approach.


### Install the necessary packages
Make sure your virtual environment is activated.
```bash
# This is what it should look like when your virtual environment is activated
(bk24tr_env) C:\Users\example>

# Running the following line will pip install all the necessary packages for the project
pip install -r requirements.txt
```

### Choose an IDE to work with, we will use Visual Studio Code in this example
1. Start VS Code
2. Click on 'Open Folder'
3. Navigate to the folder that we created earlier when we cloned the git repo 'bk24tr_hackathon'
4. You should now see the contents of the files from this repo (README.md, requirements.txt etc)
5. Now we have to select a Python interpreter that should be used to execute the code, follow the next steps
6. CTRL+SHIFT+P will open the search bar at the top center of the screen
7. Type 'Python: Select Interpreter' and click on it <br>
![image](https://github.com/jesab/bk24tr_hackathon/assets/118614390/3c7ac42a-c0a3-4a64-a5fa-d6158612427d)
8. Now you get a list of interpreters to choose from, look for your Virtual Environment bk24tr_env in the list. If you cant find it, follow next step<br>
![image](https://github.com/jesab/bk24tr_hackathon/assets/118614390/1a98a8d9-a8b5-4b57-a205-5af4edab027a)
9. ***If necessary***: If you cannot see your virtual env in the list you can click on 'Enter interpreter path...' and manually input the directory of the virtual env.
10. Make sure your cursor is inside a '.py' file and check the bottom ribbon on the right side for the following:<br>
![image](https://github.com/jesab/bk24tr_hackathon/assets/118614390/2746af1f-f816-4c5b-abe2-69dcd720d8a6)

### Inspect the .gitignore file
As you can see the content is:
```bash
.streamlit/secrets.toml
```
The secrets.toml file inside the .streamlit folder will always be ignored by git meaning it will not be commited and pushed online.
The idea is to store personal and sensitive information such as API keyes and other credentials so we can use them in our program without risking them leaking out.

### Create your local secrets file
1. Create a new file named 'secrets.toml' inside the '.streamlits/' directory
2. Once the file is created, check for it in your directory (in VS Code), it should be grayed out.<br>
![image](https://github.com/jesab/bk24tr_hackathon/assets/118614390/f7c1e417-491d-435f-883a-cd0744c20ae4)

3. Open the 'secrets.toml' file and input the following:
	***The API Keys will be provided for you when the hackathon begins***
   ```bash
   OPENAI_API_KEY = "input-the-key-that-you-recieved-here"
   PINECONE_API_KEY = "input-the-key-that-you-recieved-here"
   ...
   ```
4. ***EXAMPLE***: In our python file, we can fetch the API key from secrets by using the streamlit command:
   ```python
   # Syntax
   st.secrets["OPEN_AI_API_KEY"]

   # Example usage: store the API key in a variable using secrets
   openai_api_key = st.secrets["OPEN_AI_API_KEY"]

   # Initiate the OpenAI client so we can create Chat Completions:
   client = OpenAI(
                   api_key = openai_api_key
             )
   ```

### Running the Streamlit app
When you added all the required API keys to the secrets.toml file, you are ready to run the app.
1. Right click on the 'app.py' file in your sidebar folder navigator
2. Choose 'Open in Integrated Terminal'
3. A terminal will open at the bottom of your code editor
4. This terminal will now be in the correct directory path AND your correct virtual environment will be activated aswell.
5. Run the following line to start the Streamlit app:
	```bash
	streamlit run app.py
	```
6. The app should open in a new tab in your default browser and the terminal should indicate that the localhost server is up and running.<br>
![image](https://github.com/jesab/bk24tr_hackathon/assets/118614390/ea0217f1-6266-490b-8f73-fdb5e2b0d76a)
