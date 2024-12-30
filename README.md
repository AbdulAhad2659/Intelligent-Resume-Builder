# Intelligent Resume Builder

This project is an intelligent resume builder application that leverages Natural Language Processing (NLP) and Machine Learning (ML) to help users create professional and tailored resumes. The application interacts with users through a conversational interface, dynamically generating questions based on their profession and previous answers. It then uses this information to construct a well-formatted HTML resume.

## Features

*   **Dynamic Question Generation:**
    *   Uses advanced NLP techniques, including dependency parsing and a Transformer-based language model (T5), to generate relevant and context-aware follow-up questions.
    *   Adapts questions based on the user's profession and the ongoing conversation.
*   **Knowledge Base Integration:**
    *   Integrates with Wikidata to retrieve information about companies and other entities mentioned by the user, enhancing the context of questions.
*   **Conversation Management:**
    *   Tracks the dialogue state, including the current topic, entities discussed, and user expertise, to guide the conversation flow.
    *   Employs a rule-based dialogue policy to determine the next action (e.g., ask a question, generate the resume).
*   **Machine Learning for Categorization:**
    *   Uses a trained ML model (TF-IDF + Naive Bayes in the current implementation) to categorize user answers into predefined categories (e.g., "skills," "experience," "project").
*   **Expertise Rating:**
    *   Estimates the user's expertise level in different skills based on their answers and profession-specific skill weights.
*   **HTML Resume Generation:**
    *   Generates a well-formatted HTML resume based on the gathered information and expertise ratings.

## Prerequisites

Before running the application, make sure you have the following installed:

*   Python 3.x
*   spaCy (with the `en_core_web_lg` language model)
*   transformers
*   torch
*   scikit-learn
*   requests

You can install the dependencies using pip:

```bash
pip install spacy transformers torch scikit-learn requests
python -m spacy download en_core_web_lg
```

## Installation

1. Clone the repository:

    ```bash
    git clone <https://github.com/AbdulAhad2659/Intelligent-Resume-Builder>
    ```

2. Change to the project directory:

    ```bash
    cd <Intelligent-Resume-Builder>
    ```

3. Install the required dependencies (see Prerequisites).

## Usage

1. **Prepare your ML training data:**

    *   Replace the dummy `answer_data` in the script with a real, labeled dataset for training the text classification model. The data should be a list of tuples, where each tuple contains an answer (string) and its corresponding category label (string).

2. **Run the application:**

    ```bash
    python main.py
    ```

3. **Interact with the application:**

    *   The application will start by asking for your basic information (name, email, profession, etc.).
    *   It will then ask a series of questions, dynamically generated based on your answers.
    *   Once enough information is gathered, it will generate an `resume.html` file in the same directory.

4. **View the resume:**

    *   Open the `resume.html` file in your web browser to view the generated resume.

## Further Development

This project is a starting point and can be further developed in many ways:

*   **Improve NLP:**
    *   Fine-tune a larger Transformer model (e.g., T5-base, T5-large) on a relevant dataset for better question generation.
    *   Incorporate more advanced NLP techniques like semantic role labeling and coreference resolution.

*   **Enhance Knowledge Base Integration:**
    *   Use a dedicated Wikidata library (e.g., `qwikidata`).
    *   Query for more information and use it to enrich the conversation and resume content.
    *   Implement caching to improve performance.

*   **Advanced Conversation Management:**
    *   Develop a more sophisticated dialogue policy using machine learning (e.g., reinforcement learning).
    *   Track dialogue state more comprehensively.

*   **Robust ML Models:**
    *   Train more accurate text classification models using larger datasets and different algorithms (e.g., SVM, deep learning).
    *   Develop regression or ranking models for more accurate expertise rating.

*   **Web Application:**
    *   Build a user-friendly web interface using Flask or Django.

*   **Database Integration:**
    *   Store user data, conversation history, and generated resumes in a database.

*   **ATS Optimization:**
    *   Research ATS-friendly keywords and formatting.
    *   Provide suggestions to the user on how to improve their resume for ATS.

*   **User Authentication:**
    *   Implement user accounts and secure authentication.

*   **Testing:**
    *   Write unit tests and conduct user testing to ensure quality and usability.

## Ethical Considerations

*   **Bias:** Be mindful of potential biases in the data and models, and take steps to mitigate them.
*   **Transparency:** Be transparent with users about how their data is used and how the application works.
*   **Fairness:** Ensure the application does not disadvantage any particular group of users.

## Contributing

Contributions to this project are welcome! Please feel free to submit pull requests or open issues to discuss improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.