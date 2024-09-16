# AI Vegan Nutritionist

This repository contains a Retrieval-Augmented Generation (RAG) application that serves as an AI-powered vegan nutritionist. The application is designed to provide users with accurate and up-to-date information on vegan nutrition by leveraging scientific research papers.

<img src="App_Demo.gif" alt="Streamlit Demo" width="650"/>


## Project Overview

The AI Vegan Nutritionist is built using the following pipeline:

1. **Data Ingestion**: Research paper data is downloaded from the Springer Nature API.
2. **Data Storage**: The raw data is saved into an Amazon S3 bucket.
3. **Data Processing**: The stored data is processed and prepared for embedding.
4. **Vector Database**: Processed data is embedded and stored in an AWS vector database.
5. **User Interface**: A Streamlit app is created to allow users to interact with the AI nutritionist through a chat interface.

## Features

- Retrieves and processes the latest research on vegan nutrition
- Utilizes RAG to provide accurate and contextual responses
- Interactive chat interface for user queries
- Continuously updatable knowledge base

## Technologies Used

- Python
- Springer Nature API
- Amazon Web Services (AWS)
  - S3 for raw data storage
  - Vector database for embeddings
  - Terraform for AWS infrastructure
  - AWS Bedrock for RAG and LLMs
- Streamlit for the user interface

## Getting Started

To run this project locally, follow these steps:

1. Clone the repository
2. Install Poetry to manage dependencies
3. Set up your AWS credentials and Springer Nature API key
4. Run Terraform scripts to automatically set up AWS infrastructure
5. Run the data ingestion script to populate the S3 bucket
6. Process the data and create embeddings
7. Launch the Streamlit app

Detailed instructions for each step can be found in the project documentation.

## Usage

Once the Streamlit app is running, users can interact with the AI Vegan Nutritionist by:

1. Typing questions about vegan nutrition into the chat interface
2. Receiving responses based on the latest research in the field
3. Exploring follow-up questions and diving deeper into topics of interest

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Springer Nature for providing access to their research database
- The open-source community for the tools and libraries used in this project

For more information, please refer to the documentation in the `docs` folder.
