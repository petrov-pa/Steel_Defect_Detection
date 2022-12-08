"""This module runs the program as a flask application."""

from flask import Flask, abort, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
import os
import warnings
import cv2
import numpy as np
from src.models import FixedDropout
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from werkzeug.wrappers import Response
warnings.filterwarnings("ignore")


def run_predict(filename: str) -> None:
    """
    Return file with predicted image.

    :param filename: str
    :return: filename: str
    """
    # загружаем обученную модель
    clf = load_model(
        "../project/weights/clf.h5", custom_objects={"FixedDropout": FixedDropout},
        compile=False
    )
    linknet = load_model(
        "../project/weights/linknet.h5",
        custom_objects={"FixedDropout": FixedDropout},
        compile=False,
    )
    orig_img = cv2.imread("./data/test/" + filename)
    norm_img = orig_img / 255
    norm_img = [norm_img[:, 320 * part: 320 * (part + 1), :] for part in range(5)]
    norm_img = np.array(norm_img)
    # предсказываем наличие дефекта
    pred_clf = clf.predict(norm_img)
    if np.sum(pred_clf > 0.5):  # если дефект есть, то делаем сегментацию
        pred_seg = linknet.predict(norm_img)
        full_pred = np.hstack(
            [pred_seg[0], pred_seg[1], pred_seg[2], pred_seg[3], pred_seg[4]]
        )
        pred_mask = (full_pred > 0.5).astype(np.int32)
        pred_mask = pred_mask * 100
        pred_mask[:, :, 0] = pred_mask[:, :, 0] + pred_mask[:, :, 3]
        pred_mask = pred_mask[:, :, :-1]
    else:
        pred_mask = np.zeros((256, 1600, 1))
        # записываем в файл
    cv2.imwrite("../project/static/" + filename, orig_img + pred_mask)
    return filename


app = Flask(__name__)

app.config.update(
    {"SECRET_KEY": "sa2sd7gg4sa7a5as4d54fa78", "WTF_CSRF_SECRET_KEY": "k55h2l8o2n1n5g0"}
)
app.config["UPLOAD_FOLDER"] = "/project/outputs"


@app.route("/badrequest400")
def bad_request() -> Response:
    """Return 400 Bad Request."""
    return abort(400)


class MyForm(FlaskForm):
    """Get interface."""

    file = FileField(
        "Изображение стального листа: ",
        validators=[
            FileRequired(),
            FileAllowed(["png", "jpg", "bmp"], "Загрузите изображение"),
        ],
    )


@app.route("/", methods=("GET", "POST"))
def predict() -> Response:
    """Return the response of the service."""
    form = MyForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join("./data/test", filename))
        defect_image = run_predict(filename)
        return render_template("submit.html", form=form, user_image=defect_image)
    return render_template("submit.html", form=form, name=" ")


app.run()
