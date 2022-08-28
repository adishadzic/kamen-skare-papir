from datetime import datetime
from tensorflow.keras.models import load_model
import cv2 
import numpy as np

CLASSES = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "nothing"
}

# vraća onu klasu čiji smo joj indeks prosljedili kao parametar
def mapper(val):
    return CLASSES[val]

def determineWinner(move1, move2):
    if move1 == "" and move2 == "":
        return "Waiting..."
    else:
        if move1 == "scissors" and move2 == "paper":
                return "Player 1"
        elif move1 == "paper" and move2 == "rock":
                return "Player 1"
        elif move1 == "rock" and move2 == "scissors":
                return "Player 1"
        elif move1 == move2 :
                return "Tie"
        else:
            return "Player 2"


model = load_model("RockPaperScissor_60_DEGREE_GRAYSCALE-1.h5")
p1_move = p2_move = None
duration = 10

vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1800);
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600);
qu = 0
player1_move_name=""
player2_move_name=""
player1Score = 0
player2Score = 0

prev_winner = ""

winner = "Waiting..."
prev_winner = ""

while True:

    ret, frame = vc.read()
    start_time = datetime.now()
    diff = (datetime.now() - start_time).seconds

    while (diff <= duration):

        ret, frame = vc.read()
        frame = cv2.flip(frame, 1)

        rectangle1_start_point = (300, 500)
        rectangle1_end_point = (700, 900)

        rectangle2_start_point = (1200, 500)
        rectangle2_end_point = (1600, 900)

        rectangle_border_color = (0, 0, 255)
        rectangle_border_thickness = 2

        cv2.rectangle(frame, rectangle1_start_point, rectangle1_end_point, rectangle_border_color, rectangle_border_thickness)
        cv2.rectangle(frame, rectangle2_start_point, rectangle2_end_point, rectangle_border_color, rectangle_border_thickness)

        if 0 <= diff < 3:
            success = True
            gameText = "Ready?"
        elif diff < 5:
            gameText = "3..."
        elif diff < 6:
            gameText = "2..."
        elif diff < 7:
            gameText = "1..."
        elif diff < 8:
            gameText = "GO..."
        elif diff == 9:
            # izvlačenje okvira prvog igrača 
            roi1 = frame[500:900, 300:700]
            # mijenjanje boje slike u sive tonove
            img1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            # resizeanje na 80x80
            img1 = cv2.resize(img1, (80, 80))
            pred1 = model.predict(np.array([img1]))
            # određivanje indeksa onog elementa u arrayu koji ima najvecu vrijednost (1)
            move_code1 = np.argmax(pred1[0])
            player1_move_name = mapper(move_code1)

            # isti proces za drugog igrača
            roi2 = frame[500:900, 1200:1600]
            img2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            img2 = cv2.resize(img2, (80, 80))
            pred2 = model.predict(np.array([img2]))
            move_code2 = np.argmax(pred2[0])
            player2_move_name = mapper(move_code2)


        if player1_move_name != "nothing" and p2_move != "nothing":
            win = determineWinner(str(player1_move_name), str(player2_move_name))

        else:
            player2_move_name = "nothing"
            player1_move_name = "nothing"

            winner = "Waiting..."
        p1_move = player1_move_name
        p2_move = player2_move_name

        textFont1 = cv2.FONT_HERSHEY_SIMPLEX
        textFont2 = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, gameText, (890, 90), textFont2, 3, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.putText(frame, "Player 1: " + player1_move_name, (300, 950), textFont1, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Player 2: " + player2_move_name, (1200, 950), textFont1, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Winner: " + winner, (750, 300), textFont2, 3, (0, 150, 0), 2, cv2.LINE_AA)

        scoreColor = (34,34,178)
        cv2.putText(frame, str(player1Score), (730, 720), textFont1, 4, scoreColor, 5, cv2.LINE_AA)
        cv2.putText(frame, ":", (930, 700), textFont1, 4, scoreColor, 5, cv2.LINE_AA)
        cv2.putText(frame, str(player2Score), (1090, 720), textFont1, 4, scoreColor, 5, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        diff = (datetime.now() - start_time).seconds

        k = cv2.waitKey(10)
        if k & 0xFF == ord("r"):  
            break
        if k & 0xFF == ord("q"):  
            qu = 1
            break

    if win == "Player 1":
        player1Score += 1
        if player1Score==3:
            winner= "Player 1"
            player1Score=0
            player2Score=0
    elif win == "Player 2":
        player2Score += 1
        if player2Score==3:
            winner= "Player 2"
            player1Score = 0
            player2Score = 0
    else:
        player1Score += 0
        player2Score += 0
        winner = "Waiting..."


    if qu == 1:
        break

vc.release()
cv2.destroyAllWindows()