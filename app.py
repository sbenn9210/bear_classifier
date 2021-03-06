import streamlit as st
from fastai.vision.all import *
from pathlib import Path

path = Path('export.pkl')

@st.cache(allow_output_mutation = True)
def learner(path):
    learn = load_learner(path)
    return learn

def main():
    
    st.title("Bear Classifier")
    st.markdown("Upload an image of a bear. Disclaimer: This model only works on grizzly bears, black bears, and teddy bears. It will not be able to give you an answer for hot dog or not hot dog.")

    learn = learner(path)
    uploaded_file = st.file_uploader('Choose a image file', type=['png', "jpeg", "jpg"])
    if uploaded_file is not None:
        st.image(uploaded_file.read(), width=300)
        our_image = PILImage.create(uploaded_file)
        pred,pred_idx,probs = learn.predict(our_image)
        st.success(pred + " bear")
        st.write("Prediction: ", pred, "; Probability: ", probs[pred_idx]*100)

        



if __name__=='__main__':
    main()