from fastai.vision.all import *
import gradio as gr

def is_cat(x):
    return x[0].isupper()

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

learn=load_learner('model.pkl')
categories = ('Dog','Cat')

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['images/cat.jpg', 'images/dog.jpg', 'images/dunno.jpg']
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
