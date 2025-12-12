import pyttsx3
import time

# TEST DATA (only questions)
details = """
What is your name?
How old are you?
Why should we hire you?
"""

# -------- VOICE SETUP -------- #
engine = pyttsx3.init()
engine.setProperty("rate", 175)
engine.setProperty("volume", 1.0)

def speak(text):
    print(f"\nAI Interviewer: {text}")
    engine.say(text)
    engine.runAndWait()

def ask_questions_from_details(details_text):

    questions = []
    for line in details_text.split("\n"):
        clean = line.strip()
        if "?" in clean:      # detect questions
            questions.append(clean)

    print(f"\nTotal Questions Found: {len(questions)}\n")

    if len(questions) == 0:
        speak("No questions found in the details variable.")
        return

    for i, q in enumerate(questions, start=1):
        speak(f"Question {i}: {q}")
        print("Waiting 3 seconds for test...\n")
        time.sleep(30)  # shorter wait for testing

    speak("Interview completed.")
    

# ---- RUN TEST ---- #
ask_questions_from_details(details)
