import numpy as np
from models import FixedDropout
import cv2
import os
from tensorflow.keras.models import load_model
from flask import Flask, abort, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")


def run(filename):
    # загружаем обученную модель
    clf = load_model('./models/clf.h5', custom_objects={'FixedDropout': FixedDropout}, compile=False)
    linknet = load_model('./models/linknet.h5', custom_objects={'FixedDropout': FixedDropout}, compile=False)
    orig_img = cv2.imread('./inputs/' + filename)
    norm_img = orig_img/255
    # предсказываем наличие дефекта
    pred_clf = clf.predict(np.array([norm_img[:, 320 * part:320 * (part + 1), :] for part in range(5)]))
    if np.sum(pred_clf > 0.5):  # если дефект есть, то делаем сегментацию
        pred_seg = linknet.predict(np.array([norm_img[:, 320 * part:320 * (part + 1), :] for part in range(5)]))
        full_pred = np.hstack([pred_seg[0], pred_seg[1], pred_seg[2], pred_seg[3], pred_seg[4]])
        pred_mask = (full_pred > 0.5).astype(np.int32)
        pred_mask = pred_mask * 100
        pred_mask[:, :, 0] = pred_mask[:, :, 0] + pred_mask[:, :, 3]
        pred_mask = pred_mask[:, :, :-1]
    else:
        pred_mask = np.zeros((256, 1600, 1))
        # записываем в файл
    cv2.imwrite('./static/' + filename, orig_img+pred_mask)
    return filename


app = Flask(__name__)

app.config.update(dict(
                SECRET_KEY="sa2sd7gg4sa7a5as4d54fa78",
                WTF_CSRF_SECRET_KEY="k55h2l8o2n1n5g0"
            ))
app.config['UPLOAD_FOLDER'] = 'outputs'


@app.route('/badrequest400')
def bad_request():
    return abort(400)


class MyForm(FlaskForm):
    file = FileField('Фотография стального листа: ',
                     validators=[FileRequired(), FileAllowed(['png', 'jpg', 'bmp'], 'Загрузите изображение')])


@app.route('/', methods=('GET', 'POST'))
def predict():
    form = MyForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join('./inputs', filename))
        defect_image = run(filename)
        return render_template('submit.html', form=form, user_image=defect_image)
    return render_template('submit.html', form=form, name=' ')


app.run()
