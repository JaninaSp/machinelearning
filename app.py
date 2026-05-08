import streamlit as st

st.set_page_config(page_title="Library Book Recommender", layout="wide")

st.title("Library Book Recommender")
st.write("Welcome to our book recommendation demo.")

user_type = st.radio("Who are you?", ["Existing User", "New User"])

if user_type == "Existing User":
    user_id = st.text_input("Enter your user ID")
    if st.button("Get Recommendations"):
        st.success(f"Recommendations for user {user_id} will appear here.")

else:
    fav_subject = st.text_input("What subject do you like?")
    fav_author = st.text_input("Who is your favorite author?")
    if st.button("Generate Recommendations"):
        st.success("Starter recommendations will appear here.")