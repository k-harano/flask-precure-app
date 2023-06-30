import os
import cv2
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
import japanize_matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

precure_list = pd.read_excel("./precure_list.ods")
precure_list["No."] = precure_list["No."].astype(str).str.zfill(2)
input_size_h = 224
input_size_w = 224
model = load_model("./model.h5", compile=False)
target_layer_name = "block5_conv3"
preds_num = 1

def make_gradcam_heatmap(img, model, target_layer_name, pred_index=None):
    grad_model = tf.keras.Model(
        [model.inputs], [model.get_layer(target_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        target_layer_output, preds = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, target_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # αの計算
    
    heatmap = target_layer_output[0] @ pooled_grads[..., tf.newaxis] # α * A^k
    heatmap = tf.squeeze(heatmap)
    
    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            os.remove(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img2 = cv2.resize(img / 255., (input_size_w, input_size_h))
            x = img2.reshape(1, *img2.shape)
            preds = model.predict(x)
            preds = preds.reshape(-1)
            preds_idxs = preds.argsort()[::-1]
            
            img_c = img
            for i in range(preds_num):
                heatmap = make_gradcam_heatmap(x, model, target_layer_name, pred_index=preds_idxs[i])
                heatmap = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                img_h = cv2.addWeighted(img, 0.5, heatmap, 1 - 0.5, 0)
                img_c = cv2.hconcat([img_c, img_h])
            
            fig = plt.figure(figsize=(14, 7))
            plt.text(0, -0.08 * img.shape[0], "入力画像", fontsize="large")
            for i in range(preds_num):
                plt.text(img.shape[1] * (i + 1), -0.08 * img.shape[0], str(precure_list["No."][preds_idxs[i]]) + "." + precure_list["作品名"][preds_idxs[i]], fontsize="large")
                plt.text(img.shape[1] * (i + 1), -0.02 * img.shape[0], precure_list["キャラクター名"][preds_idxs[i]] + " ({:.2%})".format(preds[preds_idxs[i]]), fontsize="large")
            plt.imshow(img_c)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            data = base64.b64encode(buf.getbuffer()).decode("utf-8")
            return render_template('index.html', data=data)
    return render_template("index.html", data="")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)