import pyttsx3

def create_sample():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.save_to_file("Hello, how are you today?", "sample.wav")
    engine.runAndWait()
    print("sample.wav generated.")

if __name__ == "__main__":
    create_sample()