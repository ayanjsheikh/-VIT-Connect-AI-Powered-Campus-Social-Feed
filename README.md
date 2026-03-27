# 🎓 VIT Connect — AI-Powered Campus Social Feed

**VIT Connect** is a Build Your Own Project (BYOP) developed for the VIT Bhopal AI Course. It is a fully functional, personalized social media recommendation engine tailored for campus life. The application features user authentication, post creation, dynamic categorisation, and a smart feed powered by Machine Learning (Content-Based Filtering using TF-IDF and Cosine Similarity).

---

## 🚀 Features

* **User Authentication:** Secure login and registration system with profile generation.
* **Dynamic Smart Feed:** A personalized feed that learns from your interactions.
* **Campus Categories:** Filter posts by Clubs, Exams, Canteen, Events, Hostel, and General.
* **Interactive Engagement:** Like, hide (disinterest), and delete posts to train your personal AI agent.
* **Live Analytics:** Real-time profile strength, top match percentage, and an AI-generated "Interest Cloud" based on your activity.

---

## 🧠 The Intelligence Behind the App (AI Course Mapping)

This application transforms a simple chronological list of posts into a "Smart Feed" using core Artificial Intelligence concepts. Here is how the AI operates:

### 1. Perception (The "Sensors") — *CO2: Knowledge Representation*
The AI doesn't just see "text." It uses Natural Language Processing (NLP) to perceive features from the data. If a post says "Hackathon at Lab 1," the AI extracts features like *Category: Tech*, *Location: Lab*, and *Urgency: High*. Through **TF-IDF Vectorisation**, it translates human language into a mathematical knowledge representation that the system can process.

### 2. Reasoning (The "Brain") — *CO1: Rational Agent & CO3: Probability*
The AI acts as a **Rational Agent**. Its primary "Goal" is to maximize your engagement. It uses probabilistic reasoning to predict user behavior:
"Given that the user liked 3 'Coding' posts earlier, what is the Probability $P(\text{Like} | \text{Tech Post})$?" 
By calculating **Cosine Similarity** between the user's learned profile vector and the available posts, it ranks the posts with the highest mathematical probability of engagement at the top.

### 3. Learning (The "Adaptation") — *CO4: Machine Learning*
Every time you click "Like" or "Not Interested," the AI updates its internal model in real-time. It reduces the **Bias** (what it assumes you like) and adjusts the **Variance** (how much new content it exposes you to). The agent is literally learning your unique personality and adapting its multidimensional user profile vector based on your active data.

### 4. Optimization (The "Heuristic") — *CO2: Search Heuristics*
Because a real social media feed contains millions of posts, the AI utilizes Search Heuristics to quickly filter and present the best posts. Instead of exhaustively checking every single post in the database sequentially, it calculates proximity in the vector space, saving computational power and reducing latency.

---

## 🛠️ Tech Stack

* [cite_start]**Frontend & Web Framework:** Streamlit 
* [cite_start]**Data Handling:** Pandas, NumPy 
* [cite_start]**Machine Learning:** Scikit-Learn (TF-IDF Vectoriser, Cosine Similarity) 
* **Security:** Built-in `hashlib` for password hashing

---

## 💻 Installation & Usage

**1. Clone the repository and navigate to the project directory.**

**2. Install the required dependencies:**
```bash
pip install -r requirements.txt
