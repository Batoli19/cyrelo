import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 165)  # speaking speed
engine.setProperty("volume", 1.0)

engine.say("Cyrelo is online. Voice alerts are working.")
engine.runAndWait()

print("✅ Voice test done.")
