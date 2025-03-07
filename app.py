from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
import pickle
import fitz
from werkzeug.utils import secure_filename
import pymongo
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import docx2txt
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
nlp = spacy.load("en_core_web_sm")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)



def preprocess(text):
    doc = nlp(text.lower())
    keywords = []
    for token in doc:
        if not token.is_stop and token.is_alpha:
            if token.ent_type_ in ["ORG", "PRODUCT", "SKILL"] or token.text in ["java", "javascript"]:
                keywords.append(token.text)  # Preserve technical keywords
            else:
                keywords.append(token.lemma_)
    return " ".join(keywords)

def match_resume(resume_text, job_description):
    processed_resume = preprocess(resume_text)
    processed_job_desc = preprocess(job_description)

    # Extract keywords from text (tokens that are meaningful skills)
    resume_keywords = set([token.text.lower() for token in nlp(processed_resume) if token.is_alpha])
    job_keywords = set([token.text.lower() for token in nlp(processed_job_desc) if token.is_alpha])

    # Identify missing skills
    missing_skills = list(job_keywords - resume_keywords)
    
    # If all job skills are in the resume, match is 100%
    if not missing_skills:
        similarity_score = 100
    else:
        # Calculate percentage match based on skill presence
        matched_skills = len(job_keywords - set(missing_skills))
        total_skills = len(job_keywords)
        similarity_score = (matched_skills / total_skills) * 100

    return round(similarity_score, 2), missing_skills




@app.route("/match_resume", methods=["POST"])
def match_resume_api():
    data = request.get_json()
    resume_text = data.get("resume_text", "").strip()
    job_description = data.get("job_description", "").strip()

    if not resume_text or not job_description:
        return jsonify({"error": "Both resume text and job description are required!"}), 400

    similarity_score, missing_skills = match_resume(resume_text, job_description)

    response = {
        "match_score": round(similarity_score, 2),
        "missing_skills": missing_skills,
        "status": "Perfect Match!" if similarity_score > 80 else "Needs Improvement"
    }

    return jsonify(response)
@app.route("/matchjobs")
def match():
    return render_template("match.html")


# ----------------- Home Page -----------------
@app.route('/')
def index():
    return render_template('home.html')


# ----------------- Predict from Textarea -----------------
@app.route('/predict', methods=['POST'])
def predict_text():
    resume_text = request.form.get('resume_text', '').strip()

    if not resume_text:
        flash("Error: No resume text provided! Please paste your resume.", "error")
        return redirect(url_for('index'))  # Stay on the same page
    
    # Convert to vector and predict job category
    resume_vectorized = vectorizer.transform([resume_text])
    prediction = model.predict(resume_vectorized)
    prediction_list = prediction.tolist()

    session["prediction"] = prediction_list  # Store in session
    return redirect(url_for("show_prediction"))


# ----------------- Predict from File Upload -----------------
def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print("Error extracting text from PDF:", e)
    return text

@app.route("/upload", methods=["POST"])
def upload_resume():
    """Handles resume file upload and job prediction."""
    if "resume_file" not in request.files:
        flash("No file part found! Please upload a valid resume.", "error")
        return redirect(url_for('index'))

    file = request.files["resume_file"]
    
    if file.filename == "":
        flash("No selected file! Please choose a resume file.", "error")
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        # Extract text from file
        resume_text = extract_text_from_pdf(filepath)
        
        if not resume_text.strip():
            flash("Could not extract text from file! Please upload a valid resume.", "error")
            return redirect(url_for('index'))
        
        # Process the extracted text
        resume_vectorized = vectorizer.transform([resume_text])
        prediction = model.predict(resume_vectorized)  # Get first prediction
        prediction_list = prediction.tolist()
        session["prediction"] = prediction_list  # Store in session
        return redirect(url_for("show_prediction"))

    flash("Invalid file type! Only PDF, DOCX, and TXT are allowed.", "error")
    return redirect(url_for('index'))


# ----------------- Display Prediction -----------------
@app.route("/show_prediction")
def show_prediction():
    prediction = session.get("prediction", "Unknown Job Category")
    return render_template("predict.html", prediction=prediction)

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route('/terms-of-service')
def terms_of_service():
    return render_template('terms_of_service.html')

if __name__ == "__main__":
    app.run(debug=True)