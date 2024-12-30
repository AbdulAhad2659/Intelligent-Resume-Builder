import spacy
import random
import requests
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # Example classifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- Setup ---
# Load a larger spaCy model
nlp = spacy.load("en_core_web_lg")

# --- Dummy Data for ML Model Training (Replace with your actual data) ---
answer_data = [
    ("I have experience with Python, Java, and C++.", "skills"),
    ("I worked on a project to develop a new web application.", "project"),
    ("I led a team of five engineers.", "experience"),
    ("I'm proficient in using SQL and NoSQL databases.", "skills"),
    ("My expertise lies in building scalable backend systems.", "skills"),
    ("I managed a project from inception to deployment.", "project"),
    ("I have over 5 years of experience in software development.", "experience"),
    ("Proficient in cloud technologies like AWS and Azure.", "skills"),
    ("Developed a machine learning model for data analysis.", "project"),
    ("Experience in leading cross-functional teams.", "experience"),
    ("Skilled in using data visualization tools like Tableau.", "skills"),
    ("Worked on a project involving big data technologies.", "project"),
    ("Experience in financial modeling and analysis.", "experience"),
    ("Proficient in using Excel for financial analysis.", "skills"),
    ("Managed a budget of over $1 million for the project.", "project"),
    ("Experience in recruiting and onboarding new employees.", "experience"),
    ("Skilled in conflict resolution and employee relations.", "skills"),
    ("Implemented a new HR management system.", "project"),
    ("Experience in conducting market research and analysis.", "experience"),
    ("Proficient in SEO, SEM, and social media marketing.", "skills"),
    ("Managed a digital marketing campaign with a budget of $500k.", "project"),
    ("Experience in sales, consistently meeting targets.", "experience"),
    ("Skilled in using CRM software like Salesforce.", "skills"),
    ("Developed a sales strategy that increased revenue by 20%.", "project"),
    ("Experience in creating user-friendly designs.", "experience"),
    ("Proficient in design tools like Figma and Adobe XD.", "skills"),
    ("Designed a website that improved user engagement by 30%.", "project"),
    ("Experience in providing patient care in a hospital setting.", "experience"),
    ("Skilled in using electronic health record (EHR) systems.", "skills"),
    ("Administered a new treatment protocol that improved patient outcomes.", "project"),
    ("I have a strong background in user research.", "skills"),
    ("I've used design tools like Sketch and InVision.", "skills"),
    ("I was responsible for designing the user interface of the app.", "project"),
    ("I have experience in full-stack development, working with both front-end and back-end technologies.", "experience"),
    ("I am skilled in JavaScript frameworks like React and Angular.", "skills"),
    ("I contributed to an open-source project on GitHub.", "project"),
    ("I have experience in data mining and statistical analysis.", "experience"),
    ("I am proficient in R and Python for data analysis.", "skills"),
    ("I developed a predictive model for customer churn.", "project"),
    ("I have experience in managing projects using Agile methodologies.", "experience"),
    ("I am skilled in project management tools like Jira and Trello.", "skills"),
    ("I successfully delivered a project under tight deadlines.", "project"),
    ("I have experience in creating and managing digital marketing strategies.", "experience"),
    ("I am proficient in using Google Analytics and SEO tools.", "skills"),
    ("I ran an A/B test that improved conversion rates by 15%.", "project"),
    ("I have experience in conducting financial audits and risk assessments.", "experience"),
    ("I am skilled in using financial analysis software and databases.", "skills"),
    ("I prepared a financial report that helped secure funding.", "project"),
    ("I have experience in handling employee grievances and performance reviews.", "experience"),
    ("I am skilled in using HRIS systems like Workday.", "skills"),
    ("I developed a training program that improved employee performance.", "project"),
    ("I have experience in building and maintaining client relationships.", "experience"),
    ("I am skilled in negotiation and closing sales deals.", "skills"),
    ("I generated leads that resulted in a significant increase in sales.", "project"),
    ("I have experience in graphic design, including branding and marketing materials.", "experience"),
    ("I am skilled in using Adobe Creative Suite.", "skills"),
    ("I designed a logo that became the face of the brand.", "project"),
    ("I have experience in providing care in emergency situations.", "experience"),
    ("I am skilled in patient assessment and monitoring.", "skills"),
    ("I implemented a new protocol that improved response times in emergencies.", "project"),
    ("I developed a new feature for the mobile app.", "project"),
    ("I optimized the performance of the database.", "project"),
    ("I wrote unit tests to ensure code quality.", "project"),
    ("I refactored the codebase to improve maintainability.", "project"),
    ("I deployed the application to a cloud server.", "project"),
    ("I presented the project to stakeholders.", "project"),
    ("I mentored junior developers on the team.", "project"),
    ("I researched new technologies for the project.", "project"),
    ("I resolved a critical bug that was affecting users.", "project"),
    ("I created documentation for the project.", "project"),
    ("I collaborated with designers to create the user interface.", "project"),
    ("I integrated a payment gateway into the application.", "project"),
    ("I implemented security measures to protect user data.", "project"),
    ("I conducted code reviews to ensure code quality.", "project"),
    ("I used version control (Git) for the project.", "project"),
    ("I automated the build and deployment process.", "project"),
    ("I monitored the application for performance issues.", "project"),
    ("I troubleshooted and resolved production issues.", "project"),
    ("I gathered requirements from stakeholders.", "project"),
    ("I created a project plan and timeline.", "project"),
    ("I allocated resources to the project.", "project"),
    ("I tracked the progress of the project.", "project"),
    ("I managed risks and issues that arose during the project.", "project"),
    ("I communicated the project status to stakeholders.", "project"),
    ("I ensured the project was delivered on time and within budget.", "project"),
    ("I conducted user acceptance testing.", "project"),
    ("I facilitated team meetings and discussions.", "project"),
    ("I resolved conflicts within the team.", "project"),
    ("I created wireframes and mockups for the application.", "project"),
    ("I conducted user research to inform the design.", "project"),
    ("I designed the user interface and user experience.", "project"),
    ("I created a prototype of the application.", "project"),
    ("I tested the usability of the design.", "project"),
    ("I iterated on the design based on user feedback.", "project"),
    ("I created a style guide for the application.", "project"),
    ("I worked with developers to implement the design.", "project"),
    ("I ensured the design was accessible to all users.", "project"),
    ("I kept up-to-date with the latest design trends.", "project"),
    ("I created a marketing plan for the product.", "project"),
    ("I conducted market research to identify the target audience.", "project"),
    ("I developed a pricing strategy for the product.", "project"),
    ("I created advertising campaigns for the product.", "project"),
    ("I managed the social media presence of the product.", "project"),
    ("I tracked the performance of marketing campaigns.", "project"),
    ("I analyzed data to identify areas for improvement.", "project"),
    ("I presented marketing results to stakeholders.", "project"),
    ("I collaborated with sales and product teams.", "project"),
    ("I stayed up-to-date with the latest marketing trends.", "project"),
    ("I created financial models to forecast revenue.", "project"),
    ("I analyzed financial data to identify trends.", "project"),
    ("I prepared financial reports for management.", "project"),
    ("I conducted variance analysis to identify areas of concern.", "project"),
    ("I assisted in the preparation of the annual budget.", "project"),
    ("I performed financial due diligence for potential investments.", "project"),
    ("I developed financial policies and procedures.", "project"),
    ("I ensured compliance with financial regulations.", "project"),
    ("I monitored the company's financial performance.", "project"),
    ("I provided financial advice to management.", "project"),
    ("I developed a new onboarding process for new hires.", "project"),
    ("I conducted employee performance reviews.", "project"),
    ("I managed employee relations issues.", "project"),
    ("I developed and implemented HR policies.", "project"),
    ("I ensured compliance with labor laws.", "project"),
    ("I managed the recruitment and hiring process.", "project"),
    ("I administered employee benefits programs.", "project"),
    ("I developed and delivered training programs.", "project"),
    ("I conducted exit interviews.", "project"),
    ("I stayed up-to-date with the latest HR trends.", "project"),
    ("I generated leads through various channels.", "project"),
    ("I qualified leads based on their needs and budget.", "project"),
    ("I presented product demos to potential clients.", "project"),
    ("I negotiated and closed sales deals.", "project"),
    ("I managed customer relationships.", "project"),
    ("I tracked sales performance and generated reports.", "project"),
    ("I developed and implemented sales strategies.", "project"),
    ("I collaborated with marketing and product teams.", "project"),
    ("I stayed up-to-date with the latest sales techniques.", "project"),
    ("I achieved and exceeded sales targets.", "project"),
    ("I created designs for various marketing materials.", "project"),
    ("I developed brand guidelines for the company.", "project"),
    ("I designed the layout and typography for publications.", "project"),
    ("I created graphics for social media and websites.", "project"),
    ("I collaborated with marketing and product teams on design projects.", "project"),
    ("I ensured brand consistency across all designs.", "project"),
    ("I stayed up-to-date with the latest design trends.", "project"),
    ("I managed the design process from concept to completion.", "project"),
    ("I presented design concepts to stakeholders.", "project"),
    ("I incorporated feedback into design iterations.", "project"),
    ("I provided care to patients in a hospital setting.", "project"),
    ("I assessed patients' conditions and developed care plans.", "project"),
    ("I administered medications and treatments.", "project"),
    ("I monitored patients' vital signs and reported changes.", "project"),
    ("I educated patients and families about their conditions.", "project"),
    ("I collaborated with physicians and other healthcare professionals.", "project"),
    ("I documented patient care in electronic health records.", "project"),
    ("I maintained a safe and clean environment for patients.", "project"),
    ("I responded to emergencies and provided life support.", "project"),
    ("I adhered to ethical and professional standards of nursing.", "project")
]
answers, labels = zip(*answer_data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(answers, labels, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and a classifier
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB()),
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# --- Load Transformer Model and Tokenizer ---
try:
    question_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    question_model = T5ForConditionalGeneration.from_pretrained("t5-small")
except:
    print("Make sure you have transformers and torch installed")

# --- Function to Categorize Answers with the ML Model ---
def categorize_answer_with_ml(answer):
    """
    Categorizes an answer using a trained machine learning model.
    """
    category = model.predict([answer])[0]
    return category

# --- Basic Information Gathering ---

def get_basic_info():
    """Gets the user's name, contact details, and desired profession."""
    name = input("Enter your full name: ")
    email = input("Enter your email address: ")
    phone = input("Enter your phone number: ")
    linkedin = input("Enter your LinkedIn profile URL (optional): ")
    github = input("Enter your GitHub profile URL (optional): ")
    profession = input("Enter your desired profession: ").lower()
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "profession": profession,
    }

# --- Advanced NLP Techniques ---

def generate_question_with_dependency_parsing(context):
    """
    Generates follow-up questions using dependency parsing.
    """
    if not context:
        return None

    last_question, last_answer = context[-1]
    doc = nlp(last_answer)

    for token in doc:
        if token.dep_ == "dobj":
            return f"Can you tell me more about how you {' '.join([t.text for t in token.head.lefts])} {token.head.text} {token.text}?"

        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            return f"What was your role in {token.head.text}ing {token.text}?"

    return None

def generate_question_with_transformer(context):
    """
    Generates a follow-up question using a pre-trained T5 model.
    """
    if not context:
        return None

    last_question, last_answer = context[-1]
    context_str = f"question: {last_question} context: {last_answer}"

    input_ids = question_tokenizer(context_str, return_tensors="pt").input_ids
    outputs = question_model.generate(input_ids, max_length=64, num_return_sequences=1)
    generated_question = question_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_question

# --- Knowledge Base Integration (Wikidata) ---

def query_wikidata(query):
    """
    Queries Wikidata using its SPARQL endpoint.
    """
    url = "https://query.wikidata.org/sparql"
    try:
        response = requests.get(url, params={"query": query, "format": "json"})
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error querying Wikidata: {e}")
        return None

def get_company_info(company_name):
    """
    Gets information about a company from Wikidata.
    """
    query = f"""
    SELECT ?company ?companyLabel ?industryLabel ?inception
    WHERE {{
      ?company wdt:P31 wd:Q4830453;
               rdfs:label "{company_name}"@en.
      ?company wdt:P452 ?industry.
      ?company wdt:P571 ?inception.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    data = query_wikidata(query)
    if data:
        results = data["results"]["bindings"]
        if results:
            company_info = results[0]
            return {
                "company": company_info["companyLabel"]["value"],
                "industry": company_info["industryLabel"]["value"],
                "inception": company_info["inception"]["value"],
            }
    return None

def enhance_question_with_knowledge_base(question, context):
    """
    Enhances a question with information from Wikidata.
    """
    if not context:
        return question

    last_question, last_answer = context[-1]
    doc = nlp(last_answer)

    for ent in doc.ents:
        if ent.label_ == "ORG":
            company_info = get_company_info(ent.text)
            if company_info:
                question = f"Since {ent.text} is in the {company_info['industry']} industry and was founded in {company_info['inception']}, how did that context affect your work there?"
                break

    return question

# --- Conversation Management ---

def update_dialogue_state(dialogue_state, question, answer, ratings):
    """
    Updates the dialogue state based on the last turn.
    """
    dialogue_state["history"].append((question, answer))

    if "experience" in question.lower():
        dialogue_state["current_topic"] = "experience"
    elif "skills" in question.lower():
        dialogue_state["current_topic"] = "skills"

    doc = nlp(answer)
    for ent in doc.ents:
        dialogue_state["entities_discussed"].append(ent.text)

    for skill, rating in ratings.items():
        dialogue_state["user_expertise"][skill] += rating

    return dialogue_state

def get_next_action(dialogue_state, profession_data):
    """
    Determines the next action based on the dialogue state (rule-based policy).
    """
    if dialogue_state["current_topic"] == "experience":
        for ent in dialogue_state["entities_discussed"]:
            if ent in [e.text for e in nlp(dialogue_state["history"][-1][1]).ents if e.label_ == "ORG"]:
                return "ask_question", f"Can you tell me more about your experience at {ent}?"

    if sum(dialogue_state["user_expertise"].values()) < 5:
        return "ask_question", "What are some other skills you are proficient in?"

    if len(dialogue_state["history"]) >= 10:
        return "generate_resume", None

    # Use advanced NLP techniques for question generation
    question = generate_question_with_transformer(dialogue_state["history"])
    if question:
        return "ask_question", question

    question = generate_question_with_dependency_parsing(dialogue_state["history"])
    if question:
        return "ask_question", question

    # Fallback to other question generation methods if advanced NLP fails
    return "ask_question", generate_follow_up_question(dialogue_state["history"], profession_data)

# --- Enhanced Expertise Rating with Contextual Analysis ---
def rate_expertise(answers, profession_data):
    """
    Rates the user's expertise in different areas based on their answers, using a weighted keyword approach.
    """
    ratings = defaultdict(int)
    skill_weights = profession_data.get("skill_weights", {})

    for category, responses in answers.items():
        if category == "skills":
            for response in responses:
                doc = nlp(response)
                for token in doc:
                    if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                        skill = token.text.lower()
                        # Apply skill weights for more accurate rating
                        ratings[skill] += skill_weights.get(skill, 1)

    # Normalize ratings
    max_rating = max(ratings.values(), default=0)
    if max_rating > 0:
        for skill in ratings:
            ratings[skill] = 1 + int((ratings[skill] / max_rating) * 4)

    return dict(ratings)

# --- Question Generation ---
def generate_follow_up_question(context, profession_data):
    """
    Generates a follow-up question based on the conversation context.
    """
    if not context:
        return None

    last_question, last_answer = context[-1]
    doc = nlp(last_answer)

    # Use profession-specific keywords for more targeted questions
    keywords = profession_data.get("keywords", [])
    for keyword in keywords:
        if keyword in last_answer.lower():
            return f"Can you elaborate on your experience with {keyword}?"

    # Improved entity-based question generation
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "GPE", "EVENT"]:
            return f"What was your role in relation to {ent.text}?"

    # Improved keyword extraction and question formation
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in last_question.lower():
            if "experience" in last_question.lower():
                return f"How did {token.text} contribute to your experience?"
            elif any(word in last_question.lower() for word in ["skill", "proficient", "familiar"]):
                return f"Can you describe a scenario where you used {token.text}?"

    # Fallback to generic questions if specific details are not found
    generic_questions = [
        "Could you share more about a specific aspect of that?",
        "What were the key challenges you faced in that context?",
        "How did that experience shape your professional growth?"
    ]
    return random.choice(generic_questions)

# --- Profession Data ---
profession_data = {
    "software engineer": {
        "keywords": ["programming", "coding", "development", "testing", "agile", "cloud", "database", "front-end", "back-end"],
        "categories": {
            "skills": ["programming languages", "frameworks", "tools", "methodologies"],
            "experience": ["front-end", "back-end", "testing", "cloud", "database"],
            "project": ["project", "team", "development", "agile"]
        },
        "skill_weights": {"python": 3, "java": 3, "javascript": 3, "c++": 2, "sql": 2, "agile": 2, "aws": 2, "azure": 2, "gcp": 2, "react": 2, "angular": 2, "spring": 2}
    },
    "data scientist": {
        "keywords": ["data analysis", "machine learning", "statistics", "modeling", "visualization", "python", "r", "sql", "big data"],
        "categories": {
            "skills": ["programming languages", "machine learning libraries", "statistical methods", "data visualization tools"],
            "experience": ["data analysis", "machine learning", "statistical modeling", "big data"],
            "project": ["project", "analysis", "insights", "modeling"]
        },
        "skill_weights": {"python": 3, "r": 3, "sql": 2, "machine learning": 3, "statistical modeling": 3, "data visualization": 2, "hadoop": 2, "spark": 2}
    },
    "project manager": {
        "keywords": ["project management", "agile", "waterfall", "scrum", "risk management", "budget", "communication", "jira", "trello", "asana"],
        "categories": {
            "skills": ["project management methodologies", "project management software", "risk management", "budget management"],
            "experience": ["project leadership", "team management", "communication", "conflict resolution"],
            "project": ["project", "planning", "execution", "delivery"]
        },
        "skill_weights": {"agile": 3, "waterfall": 2, "scrum": 3, "risk management": 2, "budget management": 2, "jira": 2, "trello": 2, "asana": 2}
    },
    "ux/ui designer": {
        "keywords": ["user research", "wireframing", "prototyping", "user flows", "figma", "sketch", "adobe xd", "design thinking", "usability testing"],
        "categories": {
            "skills": ["design tools", "user research methods", "prototyping tools"],
            "experience": ["user research", "wireframing", "prototyping", "usability testing", "design process"],
            "project": ["project", "design", "user experience", "user interface"]
        },
        "skill_weights": {"figma": 3, "sketch": 3, "adobe xd": 3, "user research": 2, "wireframing": 2, "prototyping": 2, "usability testing": 2}
    },
    "digital marketing specialist": {
        "keywords": ["seo", "sem", "social media", "email marketing", "content marketing", "google analytics", "a/b testing", "wordpress", "digital strategy"],
        "categories": {
            "skills": ["digital marketing channels", "analytics platforms", "content management systems"],
            "experience": ["seo", "sem", "social media marketing", "email marketing", "content marketing", "a/b testing", "digital strategy"],
            "project": ["campaign", "marketing strategy", "results", "analysis"]
        },
        "skill_weights": {"seo": 3, "sem": 3, "social media marketing": 2, "email marketing": 2, "google analytics": 3, "wordpress": 2, "a/b testing": 2}
    },
    "financial analyst": {
        "keywords": ["financial modeling", "financial reporting", "budgeting", "forecasting", "valuation", "excel", "financial databases", "financial analysis"],
        "categories": {
            "skills": ["financial modeling techniques", "software tools", "financial analysis methods"],
            "experience": ["financial reporting", "budgeting", "forecasting", "financial analysis"],
            "project": ["project", "financial insights", "business decisions", "analysis"]
        },
        "skill_weights": {"financial modeling": 3, "excel": 3, "financial reporting": 2, "budgeting": 2, "forecasting": 2, "valuation": 2}
    },
    "human resources manager": {
        "keywords": ["recruitment", "onboarding", "employee relations", "performance management", "hr software", "workday", "sap successfactors", "labor laws", "hr policies"],
        "categories": {
            "skills": ["hr processes", "hr software systems", "employee relations", "performance management"],
            "experience": ["recruitment", "onboarding", "conflict resolution", "compliance", "hr management"],
            "project": ["project", "hr processes", "employee satisfaction", "hr initiatives"]
        },
        "skill_weights": {"recruitment": 3, "onboarding": 2, "employee relations": 3, "performance management": 2, "workday": 2, "sap successfactors": 2, "labor laws": 2}
    },
    "sales representative": {
        "keywords": ["sales techniques", "lead generation", "crm software", "salesforce", "hubspot", "sales forecasting", "closing deals", "client relationships"],
        "categories": {
            "skills": ["sales methodologies", "crm software", "sales techniques"],
            "experience": ["lead generation", "sales forecasting", "closing deals", "client management"],
            "project": ["sales targets", "strategies", "client relationships", "sales performance"]
        },
        "skill_weights": {"salesforce": 3, "hubspot": 3, "lead generation": 2, "sales forecasting": 2, "closing deals": 3, "client relationships": 2}
    },
    "graphic designer": {
        "keywords": ["branding", "identity design", "typography", "layout design", "adobe creative suite", "print design", "web design", "social media design"],
        "categories": {
            "skills": ["graphic design software", "design principles", "design techniques"],
            "experience": ["branding", "identity design", "typography", "layout design", "print design", "web design"],
            "project": ["design project", "creative process", "visual content", "design consistency"]
        },
        "skill_weights": {"adobe creative suite": 3, "branding": 2, "identity design": 2, "typography": 2, "layout design": 2, "print design": 2, "web design": 2}
    },
    "nurse": {
        "keywords": ["patient care", "nursing certifications", "rn", "bsn", "aprn", "ehr systems", "patient assessment", "care planning", "emergency situations"],
        "categories": {
            "skills": ["nursing certifications", "ehr systems", "patient care techniques"],
            "experience": ["patient assessment", "care planning", "emergency care", "healthcare settings"],
            "project": ["patient care", "care improvement", "patient outcomes", "healthcare initiatives"]
        },
        "skill_weights": {"rn": 3, "bsn": 3, "aprn": 3, "patient assessment": 2, "care planning": 2, "emergency care": 2, "ehr systems": 2}
    }
}

# --- Resume Generation (HTML) ---

def generate_resume_html(info, answers, ratings):
    """
    Generates an HTML resume using the gathered information.
    """

    resume_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{info['name']} - Resume</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .section {{
                margin-bottom: 20px;
            }}
            .skill-rating {{
                display: flex;
                align-items: center;
            }}
            .rating-bar {{
                width: 100px;
                height: 10px;
                background-color: #ddd;
                margin-left: 10px;
            }}
            .rating-level {{
                height: 100%;
                background-color: #4CAF50;
            }}
            .job-entry, .project-entry {{
                margin-bottom: 15px;
            }}
            .job-title, .project-title {{
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="section">
            <h1>{info['name']}</h1>
            <p>{info['email']} | {info['phone']} | {info['linkedin']} | {info['github']}</p>
        </div>

        <div class="section">
            <h2>Summary</h2>
            <p>A highly motivated {info['profession']} with expertise in various areas as mentioned below.</p>
        </div>

        <div class="section">
            <h2>Skills</h2>
            """
    for skill, rating in ratings.items():
        resume_template += f"""
            <div class="skill-rating">
                <span>{skill}</span>
                <div class="rating-bar">
                    <div class="rating-level" style="width: {rating * 20}%"></div>
                </div>
            </div>
        """

    resume_template += """
        </div>
        <div class="section">
            <h2>Experience</h2>
            """

    if 'experience' in answers:
        for experience in answers['experience']:
            resume_template += f"""
            <div class="job-entry">
                <p>{experience}</p>
            </div>
            """

    resume_template += """
        </div>

        <div class="section">
            <h2>Projects</h2>
            """

    if 'project' in answers:
        for project in answers['project']:
            resume_template += f"""
            <div class="project-entry">
                <p>{project}</p>
            </div>
            """

    resume_template += """
        </div>
    </body>
    </html>
    """

    with open("resume.html", "w") as f:
        f.write(resume_template)

    print("Resume generated as resume.html")

# --- Main Application Flow ---

if __name__ == "__main__":
    user_info = get_basic_info()
    profession = user_info["profession"]
    profession_data_entry = profession_data.get(profession, {})

    dialogue_state = {
        "current_topic": None,
        "entities_discussed": [],
        "user_expertise": defaultdict(int),
        "conversation_goal": "gather_info",
        "history": []
    }
    user_answers = {}

    initial_questions = profession_data_entry.get("initial_questions", [
        ("What are your key skills and areas of expertise?", "skills"),
        ("Describe your relevant experience in this field.", "experience")
    ])

    for question, category in initial_questions:
        answer = input(question + " ")
        category = categorize_answer_with_ml(answer)
        if category not in user_answers:
            user_answers[category] = []
        user_answers[category].append(answer)
        dialogue_state = update_dialogue_state(dialogue_state, question, answer, {})

    while dialogue_state["conversation_goal"] != "end_conversation":
        action, next_input = get_next_action(dialogue_state, profession_data_entry)

        if action == "ask_question":
            question = next_input
            question = enhance_question_with_knowledge_base(question, dialogue_state["history"])
            answer = input(question + " ")
            category = categorize_answer_with_ml(answer)
            if category not in user_answers:
                user_answers[category] = []
            user_answers[category].append(answer)
            dialogue_state = update_dialogue_state(dialogue_state, question, answer, user_ratings)
            user_ratings = rate_expertise(user_answers, profession_data_entry)  # Update ratings here

        elif action == "generate_resume":
            generate_resume_html(user_info, user_answers, user_ratings)
            dialogue_state["conversation_goal"] = "end_conversation"
