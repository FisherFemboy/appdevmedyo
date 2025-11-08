from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import threading
import os

# -----------------------------
# CONFIG
# -----------------------------
SECRET_KEY = os.environ.get("PERFORMAX_SECRET_KEY", "a-very-secret-key-that-should-be-in-env")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

SQLALCHEMY_DATABASE_URL = "sqlite:///./performax.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# -----------------------------
# DATABASE MODELS (SRS Update)
# -----------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="student")
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    records = relationship("StudentRecord", back_populates="owner", cascade="all, delete-orphan")

class StudentRecord(Base):
    __tablename__ = "student_records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # Foreign Key added
    subject = Column(String, nullable=False)
    grade = Column(Float, nullable=False)
    term = Column(String, nullable=True, default="Term 1")
    
    owner = relationship("User", back_populates="records")

# -----------------------------
# PYDANTIC SCHEMAS (SRS Update)
# -----------------------------
class Token(BaseModel):
    access_token: str
    token_type: str
    role: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserBase(BaseModel):
    username: str
    role: Literal['student', 'faculty', 'admin'] = 'student'
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserUpdate(UserBase):
    password: Optional[str] = None

class UserOut(UserBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True # Updated from orm_mode

class RecordIn(BaseModel):
    user_id: int
    subject: str
    grade: float = Field(..., gt=0, lt=101)
    term: Optional[str] = "Term 1"

class PredictRequest(BaseModel):
    grades: Dict[str, float]
    attendance_pct: Optional[float] = 100.0
    study_hours_per_week: Optional[float] = 5.0

class PredictResponse(BaseModel):
    subject_probabilities: Dict[str, float]
    overall_risk: Literal['low', 'medium', 'high']

class RecommendationResponse(BaseModel):
    recommended_tracks: List[Dict[str, Any]]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    
# -----------------------------
# APP INIT + CORS
# -----------------------------
app = FastAPI(title="Performax API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Should be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DB UTILS & AUTH HELPERS
# -----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user_by_username(db, username=username)
    if user is None: raise credentials_exception
    return user

# --- NEW/UPDATED AUTH FUNCTIONS ---
# This function will be for SHARED admin/faculty pages like "Reports"
def require_faculty_auth(current_user: User = Depends(get_current_user)):
    if current_user.role not in ['admin', 'faculty']:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin or Faculty privileges required")
    return current_user

# This function will be for ADMIN-ONLY pages like "User Management"
def require_admin_auth(current_user: User = Depends(get_current_user)):
    if current_user.role != 'admin':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return current_user
# --- END OF UPDATE ---


# -----------------------------
# ML MODEL (SRS Update: Uses real data)
# -----------------------------
MODEL_LOCK = threading.Lock()
MODELS = {}
SUBJECTS = ['HCI101', 'Networking', 'Information Management', 'Intermediate Programming']

def train_models(db: Session = SessionLocal()):
    with MODEL_LOCK:
        print("Attempting to train models...")
        records = db.query(StudentRecord).all()
        if len(records) < 10: # Don't train if there's not enough data
            print(f"Not enough data to train models (need 10, have {len(records)}). Using fallback.")
            MODELS.clear()
            return

        df = pd.DataFrame([{"user_id": r.user_id, "subject": r.subject, "grade": r.grade} for r in records])
        
        pivot = df.pivot_table(index='user_id', columns='subject', values='grade')
        pivot.fillna(pivot.mean(), inplace=True)
        
        # Add mock features for training
        np.random.seed(42)
        pivot['study_hours'] = np.random.normal(10, 3, size=len(pivot))
        pivot['attendance'] = np.random.normal(90, 5, size=len(pivot))

        # Use the global SUBJECTS list
        for s in SUBJECTS:
            if s not in pivot.columns: 
                print(f"Warning: Subject '{s}' not found in database records. Skipping model.")
                continue
            
            X = pivot[['study_hours', 'attendance', s]]
            y = (pivot[s] >= 60).astype(int) # Pass/Fail
            
            if len(y.unique()) < 2: 
                print(f"Warning: Subject '{s}' has only one outcome. Skipping model.")
                continue

            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            MODELS[s] = model
            print(f"✅ Model for '{s}' trained successfully.")
        
        db.close()

# -----------------------------
# DEMO DATA SEEDING
# -----------------------------
def seed_demo_data():
    db = SessionLocal()
    try:
        # Check if users already exist
        if db.query(User).count() == 0:
            print("--- Seeding initial demo data... ---")
            # Admin User
            admin = User(username='admin', hashed_password=get_password_hash('adminpass'), role='admin', full_name='Dr. Evelyn Cruz')
            db.add(admin)
            
            # Faculty User
            faculty = User(username='faculty1', hashed_password=get_password_hash('facultypass'), role='faculty', full_name='Prof. Ben Reyes')
            db.add(faculty)
            
            # Student Users
            s1 = User(username='student1', hashed_password=get_password_hash('studentpass'), role='student', full_name='Rafael Manzano')
            s2 = User(username='student2', hashed_password=get_password_hash('studentpass'), role='student', full_name='Juan Dela Cruz')
            db.add_all([s1, s2])
            db.commit()

            # Student Records
            records_to_add = [
                StudentRecord(user_id=s1.id, subject='HCI101', grade=88),
                StudentRecord(user_id=s1.id, subject='Networking', grade=92),
                StudentRecord(user_id=s1.id, subject='Information Management', grade=85),
                StudentRecord(user_id=s1.id, subject='Intermediate Programming', grade=90),
                
                StudentRecord(user_id=s2.id, subject='HCI101', grade=72),
                StudentRecord(user_id=s2.id, subject='Networking', grade=65),
                StudentRecord(user_id=s2.id, subject='Information Management', grade=78),
                StudentRecord(user_id=s2.id, subject='Intermediate Programming', grade=55), # At-risk grade
            ]
            db.add_all(records_to_add)
            db.commit()
            print("--- ✅ Demo data seeded. ---")
        else:
            print("--- Database already contains users. Skipping seed. ---")
    finally:
        db.close()

# -----------------------------
# APP STARTUP EVENT
# -----------------------------
@app.on_event("startup")
def on_startup():
    print("--- Server is starting up... ---")
    # Create database tables
    Base.metadata.create_all(bind=engine)
    print("--- Database tables created (if not exist). ---")
    
    # Seed the demo data
    seed_demo_data()
    
    # Start the model training thread
    threading.Thread(target=train_models, daemon=True).start()
    print("--- Model training thread started. ---")


# -----------------------------
# AUTH & USER ROUTES (SRS Update)
# -----------------------------
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "role": user.role}

@app.get("/me", response_model=UserOut)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# --- UPDATED: Uses require_admin_auth ---
@app.get("/users", response_model=List[UserOut])
def get_all_users(admin: User = Depends(require_admin_auth), db: Session = Depends(get_db)):
    return db.query(User).all()

# --- UPDATED: Uses require_admin_auth ---
@app.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(user_in: UserCreate, admin: User = Depends(require_admin_auth), db: Session = Depends(get_db)):
    if get_user_by_username(db, user_in.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user_in.password)
    new_user = User(**user_in.dict(exclude={"password"}), hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# --- UPDATED: Uses require_admin_auth ---
@app.put("/users/{user_id}", response_model=UserOut)
def update_user(user_id: int, user_in: UserUpdate, admin: User = Depends(require_admin_auth), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    update_data = user_in.dict(exclude_unset=True)
    if "password" in update_data and update_data["password"]:
        user.hashed_password = get_password_hash(update_data["password"])
    
    for key, value in update_data.items():
        if key != "password":
            setattr(user, key, value)
            
    db.commit()
    db.refresh(user)
    return user

# --- UPDATED: Uses require_admin_auth ---
@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, admin: User = Depends(require_admin_auth), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.role == 'admin': # Prevent admin from deleting themselves
        raise HTTPException(status_code=403, detail="Cannot delete an admin account")
    db.delete(user)
    db.commit()

# -----------------------------
# STUDENT RECORD ROUTES (SRS Update)
# -----------------------------
# --- UPDATED: Uses require_admin_auth (only admins can add grades) ---
@app.post("/records", status_code=status.HTTP_201_CREATED)
def add_student_record(record_in: RecordIn, admin: User = Depends(require_admin_auth), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == record_in.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"Student with ID {record_in.user_id} not found")
    
    new_record = StudentRecord(**record_in.dict())
    db.add(new_record)
    db.commit()
    
    # Retrain models in the background after adding new data
    threading.Thread(target=train_models, daemon=True).start()
    
    return {"message": "Record added successfully"}

# -----------------------------
# AI & DASHBOARD ROUTES
# -----------------------------
TRACKS = [
    {"name": "HCI (Human-Computer Interaction)", "required": ['HCI101'], "description": "Focuses on user interface (UI) and user experience (UX) design. Good for creative students who also enjoy coding."},
    {"name": "Networking", "required": ['Networking'], "description": "Focuses on computer networks and infrastructure. Good for problem-solvers interested in systems."},
    {"name": "Information Management", "required": ['Information Management'], "description": "Focuses on database design and data management. Good for organized and logical thinkers."},
    {"name": "Intermediate Programming", "required": ['Intermediate Programming', 'Networking'], "description": "A deep dive into programming and algorithms. Good for students who excel at complex problem-solving."}
]

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, current_user: User = Depends(get_current_user)):
    with MODEL_LOCK:
        subject_probs = {}
        hits = 0
        total = 0
        for subj, grade in payload.grades.items():
            model = MODELS.get(subj)
            if model is None:
                prob = grade / 100.0 # Fallback if model isn't trained
            else:
                X = np.array([[payload.study_hours_per_week, payload.attendance_pct, grade]])
                prob = float(model.predict_proba(X)[0][1])
            
            subject_probs[subj] = round(prob, 4)
            total += 1
            if prob < 0.6: hits += 1
                
        ratio = hits / max(total, 1)
        overall = "high" if ratio >= 0.6 else "medium" if ratio >= 0.25 else "low"
        
        return PredictResponse(subject_probabilities=subject_probs, overall_risk=overall)

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: PredictRequest, current_user: User = Depends(get_current_user)):
    pred_resp = predict(payload, current_user) 
    scores = []
    
    for t in TRACKS:
        reqs = t['required']
        # Calculate score based on the average probability of passing required subjects
        probs = [pred_resp.subject_probabilities.get(r, 0.5) for r in reqs if r in pred_resp.subject_probabilities]
        score = sum(probs) / len(probs) if probs else 0.5 # Default score if no required subjects are found
        
        scores.append({
            "track": t['name'], "score": round(score, 3), "description": t['description']
        })
        
    return RecommendationResponse(recommended_tracks=sorted(scores, key=lambda x: x['score'], reverse=True))

@app.get("/dashboard/summary")
def dashboard_summary(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    my_recs = db.query(StudentRecord).filter(StudentRecord.user_id == current_user.id).all()
    my_subjects = list(set(r.subject for r in my_recs))
    avg_grade = sum(r.grade for r in my_recs) / len(my_recs) if my_recs else None
    
    return {"my_average": avg_grade, "my_subjects": my_subjects}

# -----------------------------
# REPORTS & ANALYTICS ROUTES (SRS Update)
# -----------------------------
# --- UPDATED: Uses require_faculty_auth (Admin OR Faculty) ---
@app.get("/analytics/summary")
def get_analytics_summary(current_user: User = Depends(require_faculty_auth), db: Session = Depends(get_db)):
    records = db.query(StudentRecord.subject, StudentRecord.grade).all()
    if not records:
        return {"grade_distribution": {}, "pass_fail_counts": {}, "track_enrollment": []}

    df = pd.DataFrame(records, columns=['subject', 'grade'])
    
    # Grade Distribution by Subject
    grade_dist = {subj: list(df[df['subject'] == subj]['grade']) for subj in df['subject'].unique()}
    
    # Pass/Fail Counts by Subject
    df['status'] = np.where(df['grade'] >= 60, 'Pass', 'Fail')
    pass_fail = df.groupby(['subject', 'status']).size().unstack(fill_value=0).to_dict(orient='index')

    # Mock Track Enrollment Data (as this isn't stored in the DB)
    track_enrollment = [
        {"track": "HCI", "count": np.random.randint(20, 50)},
        {"track": "Networking", "count": np.random.randint(30, 60)},
        {"track": "Info Management", "count": np.random.randint(15, 40)},
        {"track": "Programming", "count": np.random.randint(25, 55)},
    ]
    
    return {
        "grade_distribution": grade_dist,
        "pass_fail_counts": pass_fail,
        "track_enrollment": track_enrollment,
    }

# -----------------------------
# CHATBOT ROUTE
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
def chatbot(request: ChatRequest, current_user: User = Depends(get_current_user)):
    msg = request.message.lower()
    if 'help' in msg:
        reply = "I am your AI Tutor. You can ask me questions about subjects like 'Networking' or 'HCI101' to get practice problems."
    elif 'hci' in msg:
        reply = "Great! Let's talk about Human-Computer Interaction. What is the difference between UI and UX?"
    elif 'networking' in msg:
        reply = "Of course. Let's cover networking. Can you describe the 7 layers of the OSI model?"
    else:
        reply = f"I'm sorry, I'm still learning. Try asking me for 'help' or about a specific subject like 'Networking'."
    return ChatResponse(reply=reply)

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    # Corrected the reloader path for Windows
    uvicorn.run("Performax_prototype_backend:app", host="127.0.0.1", port=8000, reload=True)