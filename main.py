from app_utilities.gradio_app import app

if __name__ == "__main__":
    # import cv2
    #
    # camera = cv2.VideoCapture(0)
    #
    # return_value, image = camera.read()
    # cv2.imshow("Image", image)
    # cv2.waitKey()
    # camera.release()
    # cv2.destroyAllWindows()
    app.launch(share=True)
