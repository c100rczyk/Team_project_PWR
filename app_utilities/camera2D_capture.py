import cv2
import numpy as np
from time import sleep
import argparse


def video_capture(object):
    camera = cv2.VideoCapture(object)

    while True:
        return_value, image = camera.read()
        cv2.imshow(f"Podglad kamery /dev/video{object}", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('k'):  # przy naciśnięciu klawisza "k" przechwytujemy ekran
            id = np.random.randint(low=0, high=10000)
            cv2.imwrite(f"przechwycone_obrazy/zdj_{id}.png", image)
            print("save")
        elif key == ord('q'):
            print("quit")
            break

    camera.release()
    cv2.destroyAllWindows()
    print("release and clean")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    Będzie trzeba dopasować środowisko conda do pracy z biblioteką cv2"
    Na razie użyłem osobnego środowiska z cv2 tylko dla tego skryptu
    
    Aby sprawdzić id kamery należy: 
    zainstalować : $ sudo apt install v4l-utils
    i uruchomić  : $ v4l2-ctl --list-devices
    pokaże to listę dostępnych urządzeń, które mogą przechwytywać obraz

    komenda: $ ls -l /dev/video*   pozwala także zobaczyć dostępne użądzania video
    bezpośrednio w ścieżce

    Uruchamianie skryptu: 
    $ python3 camera2d_capture.py --id 4

    """

    parser = argparse.ArgumentParser(description="Przechwytywanie zdjęć z kamery")
    parser.add_argument("--id", type=int, required=True, help="Id kamery z której chcemy przechwytywać obraz")
    args = parser.parse_args()
    video_capture(args.id)