import sys
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os
import PySimpleGUI as sg

file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


def transform_images():
    layout = [
        [sg.Image(key="-FINALIMAGE-")],
        [
            sg.Text("Main image"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
        ],
        [
            sg.Text("Style"),
            sg.Input(size=(25, 1), key="-FILE2-"),
            sg.FileBrowse(file_types=file_types),
        ],
        [sg.Button("Load Images")],
    ]
    window = sg.Window("Neutral style transfer", layout)
    while True:
        event, values = window.read()
        print('EVENT!', event, values)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Load Images":
            main_image_path = values["-FILE-"]
            if os.path.exists(main_image_path):
                main_image = load_image(values["-FILE-"])
            else:
                print('ERROR')
                sys.exit(-1)

            style_image_path = values["-FILE2-"]
            if os.path.exists(style_image_path):
                style_image = load_image(values["-FILE2-"])
            else:
                print('ERROR')
                sys.exit(-1)

            model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
            stylized_image = model(tf.constant(main_image), tf.constant(style_image))[0]

            plt.imshow(np.squeeze(stylized_image))
            plt.show()

    window.close()


transform_images()


