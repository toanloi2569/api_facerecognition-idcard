import sys
from flask import Flask, jsonify, request, render_template, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from os.path import join as osjoin
import pickle
import datetime
import numpy as np
from process_image import ProcessImage

UPLOAD_FOLDER = 'static/uploaded_file/'
PATH_DATABASE = osjoin(os.getcwd(), 'static', 'database')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

print("Load face emb")
if not os.path.isfile(osjoin(PATH_DATABASE, 'x_vector.pkl')) or not os.path.isfile(osjoin(PATH_DATABASE, 'x_name.pkl')):
    if not os.path.isfile(osjoin(PATH_DATABASE, 'x_name_map.pkl')):
        x_vector = []
        x_name = []
        x_name_map = []
else:
    with open(osjoin(PATH_DATABASE, 'x_vector.pkl'), 'rb') as f:
        x_vector = pickle.load(f)
    with open(osjoin(PATH_DATABASE, 'x_name.pkl'), 'rb') as f:
        x_name = pickle.load(f)
    with open(osjoin(PATH_DATABASE, 'x_name_map.pkl'), 'rb') as f:
        x_name_map = pickle.load(f)

process_image = ProcessImage()


@app.route('/searchImg', methods=['GET', 'POST'])
def upload_file():
    img = request.get_json()['data']
    img = process_image.base642img(img)
    image_nearest_faces = []

    print (len(x_vector))
    print (len(x_name))
    print (len(x_name_map))

    align = face_detect(img)
    if align is None:
        error = {
            'error': 'Có lỗi xảy ra. Có thể do hệ thống không nhận diện được khuôn mặt trong ảnh. Mời nhập một ảnh khác'
        }
        return jsonify(error), 404

    name_of_nearest_faces, path_to_nearest_faces, acc_with_nearest_faces = search(align)
    print (path_to_nearest_faces)
    for i, path in enumerate(path_to_nearest_faces):
        nearest_face = process_image.load_img(path)
        image_nearest_faces.append(process_image.img2base64(nearest_face))

    response = {
        'name_of_nearest_faces': name_of_nearest_faces,
        'image_nearest_faces': image_nearest_faces,
        'acc_with_nearest_faces': acc_with_nearest_faces
    }
    return jsonify(response), 200


@app.route('/insertRecord', methods=['GET', 'POST'])
def insert_record():
    img = request.get_json()['data']
    ID = request.get_json()['ID']

    img = process_image.base642img(img)[...,::-1]
    # from matplotlib import pyplot as plt 
    # plt.imshow(img)
    # plt.show()
    align = face_detect(img)
    # from matplotlib import pyplot as plt 
    # plt.imshow(align)
    # plt.show()
    v = process_image.img2vect(align)

    path_save_img = osjoin(PATH_DATABASE, ID + '.jpeg')
    process_image.save_img(img, path_save_img)

    x_vector.append(v)
    x_name.append(ID)
    x_name_map.append(path_save_img)
    save_data()
    return jsonify('ok'), 200

def face_detect(img):
    box = process_image.get_max_box(img)
    align = process_image.align_image(img, box)
    return align

def search(align):
    yv = process_image.img2vect(align)

    list_dist = []
    list_acc = []
    for i, xv in enumerate(x_vector):
        list_dist.append(np.sum(np.square(xv - yv)))
        list_acc.append( round((2 - list_dist[i])/2,2)*100 )
    nearest_dist = np.array(list_dist).argsort()[:6]
    name_of_nearest_faces = [x_name[i] for i in nearest_dist]
    path_to_nearest_faces = [x_name_map[i] for i in nearest_dist]
    acc_with_nearest_faces = [list_acc[i] for i in nearest_dist]

    return (name_of_nearest_faces, path_to_nearest_faces, acc_with_nearest_faces)

def save_data():
    print("Save face emb")
    with open(osjoin(PATH_DATABASE, 'x_vector.pkl'), 'wb') as f:
        pickle.dump(x_vector, f)
    with open(osjoin(PATH_DATABASE, 'x_name.pkl'), 'wb') as f:
        pickle.dump(x_name, f)
    with open(osjoin(PATH_DATABASE, 'x_name_map.pkl'), 'wb') as f:
        pickle.dump(x_name_map, f)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000,
                        type=int, help='port listening')
    args = parser.parse_args()
    port = args.port
    app.secret_key = 'super secret key'
    app.run(host='0.0.0.0', port=port, debug=True)
