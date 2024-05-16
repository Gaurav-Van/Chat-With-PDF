# Chat-With-PDF

This application is designed to facilitate interaction with multiple PDF documents through a chat interface. It allows users to upload PDFs and ask questions about the content within these documents. The system uses a combination of natural language processing and vector space modeling to provide relevant responses.

Web app deployed on Streamlit Cloud: https://chat-with-pdf-gj.streamlit.app/

Demo on Youtube: https://youtu.be/bRILkTmaRng?feature=shared

Sped up Demo on Youtube: https://youtu.be/aFvL7Z36KaA?feature=shared

## Overview

The code is structured into several functions, each handling a specific part of the process:

- `get_single_pdf_chunks`: Extracts text from a single PDF and splits it into manageable chunks.
- `get_all_pdfs_chunks`: Processes multiple PDFs and aggregates the text chunks.
- `get_vector_store`: Creates a vector store from the text chunks using embeddings.
  
  ` embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")` : This line initializes an embedding model, which is used to convert text into embeddings. The GoogleGenerativeAIEmbeddings likely refers to a pre-trained model provided by Google that can generate embeddings for text.
  
  `vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)` : This line takes the text chunks, uses the embeddings model to convert them into embeddings, and then stores these embeddings in a vector store using FAISS. FAISS is a library for efficient similarity search and clustering of dense vectors
  
  `relevant_content = vectorstore.similarity_search(user_query, k=10)` : This line takes the user’s query, converts it into an embedding using the same model, and then performs a similarity search in the vector store to find the top k (in this case, 10) most similar text chunks. It retrieves the relevant content based on the similarity to the user’s query.

- `get_response`: Generates a response based on the context and question provided by the user.
- `working_process`: Orchestrates the interaction with the user and manages the chat history.
- `main`: Initializes the application, loads environment variables, and sets up the Streamlit interface.

## Dependencies

The application relies on several external libraries:

- **os**: To interact with the operating system.
- **streamlit**: To create the web interface.
- **google.generativeai**: To access Google's generative AI models.
- **PyPDF2**: To read PDF files.
- **dotenv**: To manage environment variables.
- **langchain_core**: To handle messages and chat sessions.
- **langchain.text_splitter**: To split text into chunks.
- **langchain_community.vectorstores.faiss**: To create and manage the vector store.
- **langchain_google_genai**: To use Google's generative AI embeddings.

## How It Works

1. **PDF Processing**: The user uploads PDF documents. The application reads these documents, extracts the text, and splits it into chunks.
2. **Vector Store Creation**: It then converts these text chunks into vectors using embeddings and stores them in a vector store.
3. **Chat Interface**: The user interacts with the application through a chat interface, asking questions about the PDF content.
4. **Response Generation**: The application searches the vector store for content relevant to the user's query and uses a generative AI model to formulate a response.
5. **Chat History Management**: The chat history is maintained in the session state, allowing for context-aware conversations.

## **RAG (Retrieval-Augmented Generation)**

This code represents the **RAG (Retrieval-Augmented Generation)** concept, which is a machine learning approach that combines retrieval and generation techniques to produce relevant and coherent responses.

In the RAG framework, the retrieval component (in this case, the vector store) is used to retrieve relevant information from a large corpus (the PDF documents) based on the user's query. The generation component (the generative AI model) then takes this retrieved information and generates a final response.

The key steps in the RAG process, as represented in this code, are:

1. **Retrieval**: The `get_vector_store` function creates a vector store from the text chunks, allowing for efficient retrieval of relevant information based on the user's query.
2. **Generation**: The `get_response` function uses the retrieved information from the vector store and the user's query to generate a response using a generative AI model (in this case, Google's generative AI models).

This combination of retrieval and generation allows the system to provide relevant and coherent responses to user queries based on the content of the uploaded PDF documents.

For visual representations of the RAG concept, you can refer to the following links:

- [RAG Architecture Diagram](https://github.com/Gaurav-Van/Chat-With-PDF/assets/50765800/ed380fc4-da03-4db3-983b-236d348c2688)

## How to Run It Locally

To run this application locally, follow these steps:

1. **Clone the Repository**: Download the code to your local machine.
2. **Create Virtual Env**: `python -m venv <environment_name`
3. **Install Dependencies**: Use pip to install the required libraries listed in the requirements.txt file. `pip install -r requirements.txt`
4. **Set Up Environment Variables**: Create a `.env` file with your Google API key as `GOOGLE_API_KEY`. Replace st.secrets with os.getenv
5. **Start the Application**: Run the `main` function to start the Streamlit server.
6. **Access the Interface**: Open your web browser and go to http://localhost:8501 to interact with the application.

## Usage

- **Upload PDFs**: Drag and drop your PDF files into the uploader in the sidebar.
- **Submit**: Click the 'Submit' button to process the PDFs.
- **Ask Questions**: Use the chat input to ask questions about the content of your PDFs.
- **Receive Responses**: The AI will provide answers based on the content of the uploaded PDFs.
