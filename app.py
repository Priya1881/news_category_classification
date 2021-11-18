import streamlit as st  ## streamlit
from news_prepreprocessing import clean
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def predict_category(text):
    cleaned_text = clean(text)
    tfidf=TfidfVectorizer()
    tfidf = tfidf.fit_transform([cleaned_text])
    model = pickle.load(open('model_lr.pkl', 'rb'))
    print(tfidf)
    prediction = model.predict([cleaned_text])[0]
    print(prediction)
    return (prediction)

def run():
    st.sidebar.info('You can either enter the news item online in the textbox or upload a txt file')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Txt file"))
    st.title("News Classification")
    if add_selectbox == "Online":
        text1 = st.text_area('Enter text')

        if st.button("Predict"):
            category= predict_category(text1)
            st.success("The news item belongs to " + category)

    elif add_selectbox == "Txt file":
        file_buffer = st.file_uploader("Upload text file for new item", type=["txt"])

        if st.button("Predict"):
                text_news = pd.read_csv('news.txt')
                cleaned_text=text_news['Text'].apply(clean)
                tfidf = TfidfVectorizer()
                tfidf = tfidf.fit_transform(cleaned_text)
                model = pickle.load(open('model_lr.pkl', 'rb'))
                predictions = model.predict(cleaned_text)
                predictions_df =pd.DataFrame(predictions,columns=["Category"])
                predictions_df.to_csv('predictions.txt',index=False)
                st.success("The predictions are below, to download click the Download button")
                st.dataframe(predictions_df)
                st.download_button(label='Download',data=predictions_df.to_csv(),mime='text/csv')
                #st.balloons()


if __name__ == "__main__":
    run()