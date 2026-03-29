# 🎓 VIT Connect — AI-Powered Campus Social Feed

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.0-orange.svg)
![Status](https://img.shields.io/badge/Status-Prototype-success.svg)

**VIT Connect** is a Build Your Own Project (BYOP) developed for the VIT Bhopal Artificial Intelligence Course. It is a fully functional, personalized social media recommendation engine tailored specifically for campus life.

The application features a custom-built UI, user authentication, dynamic post creation, and an intelligent "Smart Feed" powered by **Content-Based Filtering** (TF-IDF Vectorisation & Cosine Similarity). 

---

## 📑 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Core Features](#-core-features)
3. [The AI Engine (Course Mapping)](#-the-ai-engine-course-mapping)
4. [Technical Architecture](#-technical-architecture)
5. [Installation & Setup](#-installation--setup)
6. [Usage Guide & Demo](#-usage-guide--demo)
7. [Future Enhancements](#-future-enhancements)

---

## 🎯 Problem Statement
On a busy campus, students are bombarded with information across multiple WhatsApp groups, emails, and notice boards. Relevant updates (like a hackathon for CS students or a study group for CAT aspirants) often get lost in the noise. 

**Solution:** VIT Connect consolidates campus updates into unified categories (Clubs, Exams, Canteen, Events, Hostel, General) and uses a Machine Learning recommendation agent to learn what each individual user cares about, filtering the noise and highlighting relevant content.

---

## ✨ Core Features

### 🔐 User Management & UI
* **Custom Authentication:** Secure login and sign-up flows using `hashlib` SHA-256 encryption for passwords.
* **Dynamic Avatars:** Automatically generated avatars with color-coding based on username hashing.
* **Modern UI/UX:** Custom CSS implementation overriding default Streamlit styles for a modern, app-like feel (glassmorphism, gradient text, hover animations).

### 📱 Social Engagement
* **Post Creation:** Users can publish posts with titles, descriptions, categories, and expressive emojis.
* **Categorised Feeds:** Filter the timeline by specific campus interests (e.g., Canteen, Exams, Clubs).
* **Interactive Feedback Loop:** Users can "Like" (❤️) or mark posts as "Not Interested" (👎). This explicit feedback directly trains their personal AI model in real-time.

### 🧠 Analytics & Profiling
* **Live Stats Ribbon:** Real-time metrics showing total feed posts, profile strength, and top match percentages.
* **AI Interest Cloud:** A dynamically generated tag cloud (e.g., "AI", "Hackathon", "Dosa") revealing the exact keywords the Machine Learning model has associated with the user's profile.

---

## 🤖 The AI Engine (Course Mapping)

This project explicitly demonstrates core AI concepts and maps them to standard Course Outcomes (CO).

### 1. Perception & Knowledge Representation (CO2)
The AI perceives raw text data and converts it into a mathematical format it can understand. 
* **Implementation:** The `build_tfidf(posts)` function concatenates the Title, Body, and Category of every post.
* **Algorithm:** `scikit-learn`'s `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) extracts unigrams and bigrams, removing stop words, to create a sparse document matrix. 

### 2. The Rational Agent (CO1)
The application assigns a unique `RecommendationAgent` object to every logged-in user. 
* **State:** The agent maintains a `user_profile` (an N-dimensional numpy array initialized to zeros).
* **Action:** When a user likes a post, the agent executes `like_post()`, adding the specific TF-IDF vector of that post to the user's overarching profile vector.

### 3. Machine Learning & Probability (CO4 & CO3)
The system uses **Content-Based Filtering** to adapt to the user.
* **Scoring:** The agent's `rank()` method calculates the **Cosine Similarity** between the user's learned profile vector and the entire matrix of available posts.
* **Ranking:** Posts are assigned a "Relevance Score" (0% to 100%) based on the similarity cosine. The feed dynamically sorts posts so that items with the highest probability of engagement appear first. Highly relevant posts receive a "🔥 Match" badge.

---

## ⚙️ Technical Architecture

The app is built as a single-page stateful web application using Python.

* [cite_start]**Frontend & State:** `streamlit` (managing page navigation, session states, and UI rendering).
* **Data Processing:** `pandas` for handling feed dataframes and sorting logic; [cite_start]`numpy` for vector math.
* **In-Memory Database:** Uses `st.session_state` to simulate a database. It initializes with a seed dictionary of 20 pre-written campus posts (`SEED_POSTS`) and a default admin user.

---

## 🚀 Installation & Setup

Follow these steps to run the application on your local machine.

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/vit-connect-byop.git](https://github.com/yourusername/vit-connect-byop.git)
cd vit-connect-byop
