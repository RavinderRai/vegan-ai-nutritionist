# AI Vegan Nutritionist

This repository contains a Retrieval-Augmented Generation (RAG) application that serves as an AI-powered vegan nutritionist. It provides accurate and up-to-date information on vegan nutrition using scientific research papers. The AI uses LLMs to understand queries, retrieve relevant literature, and generate informative responses, ensuring users receive current and reliable information. See the demo below.

<img src="App_Demo.gif" alt="Streamlit Demo" width="650"/>

## Project Overview

The AI Vegan Nutritionist is built using the following pipeline:

1. **Data Ingestion**: Research paper data is downloaded from the Springer Nature API.
2. **Data Storage**: The raw data is saved into an Amazon S3 bucket.
3. **Data Processing**: The stored data is processed and prepared for embedding.
4. **Vector Database**: Processed data is embedded and stored in an AWS vector database.
5. **Model Fine-Tuning**: Synthetic data is generated using OpenAI GPT-4 to fine-tune the Falcon-7B Instruct model.
6. **Model Training**: The fine-tuning job is run on SageMaker.
7. **Model Deployment**: The fine-tuned model is deployed as an endpoint on SageMaker.
8. **User Interface**: A Streamlit app is created to allow users to interact with the AI nutritionist through a chat interface. Users can select whether to chat with the fine-tuned model or a larger LLM.

## Features

- Retrieves and processes the latest research on vegan nutrition
- Utilizes RAG to provide accurate and contextual responses
- Interactive chat interface for user queries
- Option to select between the fine-tuned model and a larger LLM
- Continuously updatable knowledge base

## Project Structure


## Technologies Used

- Python
- Springer Nature API
- OpenAI GPT-4 for synthetic data generation
- Amazon Web Services (AWS)
  - S3 for raw data storage
  - SageMaker for model training and deployment
  - Vector database for embeddings
  - Terraform for AWS infrastructure
  - AWS Bedrock for RAG and LLMs
- Streamlit for the user interface

## Getting Started

To run this project locally, follow these steps:

1. Clone the repository
2. Install Poetry to manage dependencies or use a virtual env with the requirements in the root
3. Set up your AWS credentials and Springer Nature API key
4. Run Terraform scripts to automatically set up AWS infrastructure
5. Run the data ingestion script to populate the S3 bucket
6. Process the data and create embeddings
7. Fine-tune the Falcon-7B Instruct model using the synthetic data
8. Deploy the fine-tuned model as an endpoint on SageMaker
9. Launch the Streamlit app

Detailed instructions for each step can be found in the project documentation.

## Usage

Once the Streamlit app is running, users can interact with the AI Vegan Nutritionist by:

1. Typing questions about vegan nutrition into the chat interface
2. Selecting whether to chat with the fine-tuned model or a larger LLM
3. Receiving responses based on the latest research in the field

## Acknowledgments

- Springer Nature for providing access to their research database
- The open-source community for the tools and libraries used in this project
