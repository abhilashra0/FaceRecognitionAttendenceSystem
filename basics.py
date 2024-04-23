import tkinter as tk
from tkinter import messagebox as mess
from tkinter import ttk
import os
import cv2
import csv
import numpy as np
from PIL import Image
import pandas as pd
import time
from datetime import datetime

# Path to the Haar cascade file for face detection
cascade_path = "haarcascade_frontalface_default.xml"

# Directory paths for storing captured images and trained model
image_dir = "TrainingImage"
model_path = "TrainedModel/trained_model.xml"



# Define the main window
window = tk.Tk()
window.title("Face Recognition Based Attendance System")
window.geometry("1280x720")
window.configure(background='#355454')

# Define functions
def on_closing():
    if mess.askyesno("Quit", "You are exiting window. Do you want to quit?"):
        window.destroy()

def about():
    mess.showinfo("About", "This Attendance System is designed by Meet Suvariya")

def clear():
    txt.delete(0, 'end')
    txt2.delete(0, 'end')
    message1.configure(text="1) Take Images  ===> 2) Save Profile")



# Function to capture images of students
def TakeImages():
    student_id = txt.get()
    student_name = txt2.get()

    # Create directories if they don't exist
    person_dir = os.path.join(image_dir, student_id)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    # Initialize the camera
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cascade_path)

    sample_num = 0  # Counter for captured images
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            sample_num += 1
            # Save the captured face image with student ID and sample number
            img_name = f"img{sample_num}.jpg"
            img_path = os.path.join(person_dir, img_name)
            cv2.imwrite(img_path, gray[y:y+h, x:x+w])
            cv2.imshow('Capturing Images', img)

        # Break the loop after capturing desired number of images
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        elif sample_num >= 30:  # Capture 30 images per student
            break

    cam.release()
    cv2.destroyAllWindows()

    # Update message
    message1.configure(text=f"Images Taken for ID: {student_id}")
    with open('StudentDetails/StudentDetails.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([len(os.listdir(image_dir)), student_id, student_name])


# Function to train the face recognition model
def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cascade_path)

    # Function to get images and labels
    def get_images_and_labels():
        image_paths = [os.path.join(image_dir, d, f) for d in os.listdir(image_dir) for f in os.listdir(os.path.join(image_dir, d))]
        face_samples = []
        ids = []

        for image_path in image_paths:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face_id = int(os.path.basename(os.path.dirname(image_path)))
            face_samples.append(img)
            ids.append(face_id)

        return face_samples, np.array(ids)

    faces, ids = get_images_and_labels()

    recognizer.train(faces, ids)
    recognizer.save(model_path)

    message1.configure(text="Profile Saved Successfully")





# Function to track attendance
def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    detector = cv2.CascadeClassifier(cascade_path)

    # Load student details
    student_details = {}
    with open('StudentDetails/StudentDetails.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        # Load student details
        for row in reader:
            student_id = int(row[1])  # Extracting ID from the second column
            student_name = row[2].strip()  # Extracting name from the third column and removing any leading/trailing whitespace
            student_details[student_id] = student_name


    # Initialize video capture
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Recognize the face
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 100:
                student_name = student_details.get(id_, "Unknown")
                confidence = f"{round(100 - confidence)}%"
                
            else:
                student_name = "Unknown"
                confidence = f"{round(100 - confidence)}%"

            cv2.putText(img, student_name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x+100, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Taking Attendance', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    mark_attendance(id_,student_name)
    date = datetime.now().strftime('%Y-%m-%d')
    show_attendence(date)


    cam.release()
    cv2.destroyAllWindows()

def show_attendence(date):
    with open(r"Attendance\Attendance_" + date +".csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        next(reader1)
        for row in reader1:
            tb.insert('',0,text=str(row[2]),values=(str(row[3]), str(row[0]),str(row[1])))


def mark_attendance(student_id, student_name):
    date = datetime.now().strftime('%Y-%m-%d')
    time_stamp = datetime.now().strftime('%H:%M:%S')
    attendance_file = "Attendance/Attendance_"+date+".csv"
    attendance_exists = os.path.exists(attendance_file)

    with open(attendance_file, 'a', newline='') as csvfile:
        fieldnames = ['Date', 'Time', 'Student_ID', 'Student_Name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not attendance_exists:
            writer.writeheader()  # Write header if file is empty

        writer.writerow({'Date': date, 'Time': time_stamp, 'Student_ID': student_id, 'Student_Name': student_name})





message3 = tk.Label(window, text="Face Recognition Based Attendance System", fg="white", bg="#355454", width=60, height=1, font=('times', 29, 'bold'))
message3.place(x=10, y=10)

frame1 = tk.Frame(window, bg="white")
frame1.place(relx=0.11, rely=0.15, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="white")
frame2.place(relx=0.51, rely=0.15, relwidth=0.39, relheight=0.80)

lbl = tk.Label(frame1, text="Enter ID", width=20, height=1, fg="black", bg="white", font=('times', 17, 'bold'))
lbl.place(x=0, y=55)

txt = tk.Entry(frame1, width=32, fg="black", bg="#e1f2f2", font=('times', 15))
txt.place(x=55, y=88)

lbl2 = tk.Label(frame1, text="Enter Name", width=20, fg="black", bg="white", font=('times', 17, 'bold'))
lbl2.place(x=0, y=140)

txt2 = tk.Entry(frame1, width=32, fg="black", bg="#e1f2f2", font=('times', 15))
txt2.place(x=55, y=173)

message1 = tk.Label(frame1, text="1) Take Images  ===> 2) Save Profile", bg="white", fg="black", width=39, height=1, font=('times', 15))
message1.place(x=7, y=300)

message = tk.Label(frame1, text="", bg="white", fg="black", width=39, height=1, font=('times', 16))
message.place(x=7, y=500)

takeImgBtn = tk.Button(frame1, text="Take Images", command=TakeImages, fg="black", bg="#00aeff", width=34, height=1, font=('times', 16))
takeImgBtn.place(x=30, y=350)

trainImgBtn = tk.Button(frame1, text="Save Profile", command=TrainImages, fg="black", bg="#00aeff", width=34, height=1, font=('times', 16))
trainImgBtn.place(x=30, y=430)

trackImgBtn = tk.Button(frame2, text="Take Attendance", command=TrackImages, fg="black", bg="#00aeff", height=1, font=('times', 16))
trackImgBtn.place(x=30, y=60)

quitBtn = tk.Button(frame2, text="Quit", command=window.destroy, fg="white", bg="#13059c", height=1, font=('times', 16))
quitBtn.place(x=30, y=450)

#Attandance table----------------------------
style = ttk.Style()
style.configure("mystyle.Treeview", highlightthickness=0, bd=0, font=('Calibri', 11)) # Modify the font of the body
style.configure("mystyle.Treeview.Heading",font=('times', 13,'bold')) # Modify the font of the headings
style.layout("mystyle.Treeview", [('mystyle.Treeview.treearea', {'sticky': 'nswe'})]) # Remove the borders
tb= ttk.Treeview(frame2,height =13,columns = ('name','date','time'),style="mystyle.Treeview")
tb.column('#0',width=82)
tb.column('name',width=130)
tb.column('date',width=133)
tb.column('time',width=133)
tb.grid(row=2,column=0,padx=(0,0),pady=(150,0),columnspan=4)
tb.heading('#0',text ='ID')
tb.heading('name',text ='NAME')
tb.heading('date',text ='DATE')
tb.heading('time',text ='TIME')

date = datetime.now().strftime('%Y-%m-%d')
show_attendence(date)

# Other GUI elements and functionalities can be added similarly

window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()